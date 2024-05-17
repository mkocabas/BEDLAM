
import os
import cv2
import joblib
import torch
import tqdm
from loguru import logger
import numpy as np
from . import constants
from multi_person_tracker import MPT
from torchvision.transforms import Normalize
from glob import glob
from train.utils.train_utils import load_pretrained_model
from train.utils.vibe_image_utils import get_single_image_crop_demo
from torch.utils.data import DataLoader

from .config import update_hparams
from ..models.hmr import HMR
from ..models.head.smplx_cam_head import SMPLXCamHead
from ..utils.renderer_cam import render_image_group

from ..utils.image_utils import crop
from ..dataset.inference import Inference


class Tester:
    def __init__(self, args):
        self.args = args
        self.model_cfg = update_hparams(args.cfg)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        self.bboxes_dict = {}

        self.model = self._build_model()
        self.smplx_cam_head = SMPLXCamHead(img_res=self.model_cfg.DATASET.IMG_RES).to(self.device)
        self._load_pretrained_model()
        self.model.eval()

    def _build_model(self):
        self.hparams = self.model_cfg
        model = HMR(
            backbone=self.hparams.MODEL.BACKBONE,
            img_res=self.hparams.DATASET.IMG_RES,
            pretrained_ckpt=self.hparams.TRAINING.PRETRAINED_CKPT,
            hparams=self.hparams,
        ).to(self.device)
        return model

    def _load_pretrained_model(self):
        # ========= Load pretrained weights ========= #
        logger.info(f'Loading pretrained model from {self.args.ckpt}')
        ckpt = torch.load(self.args.ckpt)['state_dict']
        load_pretrained_model(self.model, ckpt, overwrite_shape_mismatch=True, remove_lightning=True)
        logger.info(f'Loaded pretrained weights from \"{self.args.ckpt}\"')

    def run_tracking(self, image_folder, min_num_frames=10):
        # ========= Run tracking ========= #
        logger.info(f'Running tracking on {image_folder}...')
        # run multi object tracker
        mot = MPT(
            device=self.device,
            batch_size=self.args.tracker_batch_size,
            display=False,
            detector_type=self.args.detector,
            output_format='dict',
            yolo_img_size=self.args.yolo_img_size,
        )
        tracking_results = mot(image_folder)

        # remove tracklets if num_frames is less than MIN_NUM_FRAMES
        for person_id in list(tracking_results.keys()):
            if tracking_results[person_id]['frames'].shape[0] < min_num_frames:
                del tracking_results[person_id]

        return tracking_results
    
    def run_detector(self, all_image_folder):
        # run multi object tracker
        mot = MPT(
            device=self.device,
            batch_size=self.args.tracker_batch_size,
            display=False,
            detector_type=self.args.detector,
            output_format='dict',
            yolo_img_size=self.args.yolo_img_size,
        )
        bboxes = []
        for fold_id, image_folder in enumerate(all_image_folder):
            bboxes.append(mot.detect(image_folder))

        return bboxes

    def load_yolov5_bboxes(self, all_bbox_folder):
        # run multi object tracker
        for fold_id, bbox_folder in enumerate(all_bbox_folder):
            for bbox_file in os.listdir(bbox_folder):
                bbox = np.loadtxt(os.path.join(bbox_folder, bbox_file))
                fname = os.path.join('/'.join(bbox_folder.split('/')[-3:-1]),bbox_file.replace('.txt','.png'))
                self.bboxes_dict[fname] = bbox

    @torch.no_grad()
    def run_on_video(self, image_folder, tracking_results, output_folder, 
                     save_results=True, render_results=False, cam_intrinsics_file=None):
        # ========= Run PARE on each person ========= #
        logger.info(f'Running BEDLAM-CLIFF on each tracklet...')

        cam_intrinsics = np.loadtxt(cam_intrinsics_file) if cam_intrinsics_file is not None else None
        
        for person_id in tqdm.tqdm(list(tracking_results.keys())):
            bboxes = None

            bboxes = tracking_results[person_id]['bbox']
            frames = tracking_results[person_id]['frames']

            dataset = Inference(
                image_folder=image_folder,
                frames=frames,
                bboxes=bboxes,
                crop_size=self.model_cfg.DATASET.IMG_RES,
                return_dict=True,
                normalize_fn=self.normalize_img,
            )

            bboxes = dataset.bboxes
            frames = dataset.frames

            dataloader = DataLoader(dataset, batch_size=self.args.batch_size, num_workers=8)

            smplx_pose = []
            smplx_betas = []
            smplx_trans = []
            cam_focal_l = []
            imgnames = []
            cam_center = []
            cam_weak_persp = []
            if render_results:
                smplx_verts = []

            for batch in tqdm.tqdm(dataloader):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                        
                batch_size = batch['img'].shape[0]
                
                img_w = batch['img_w']
                img_h = batch['img_h']
                
                if cam_intrinsics is not None:
                    focal_length = torch.tensor([(cam_intrinsics[0], cam_intrinsics[1])]*batch_size).to(batch['img'])
                    camera_center = torch.tensor([(cam_intrinsics[2], cam_intrinsics[3])]*batch_size).to(batch['img'])
                    batch['img_w'] = camera_center[:, 0] * 2.0
                    batch['img_h'] = camera_center[:, 1] * 2.0
                else:
                    focal_length = (img_w * img_w + img_h * img_h) ** 0.5
                    focal_length = focal_length.repeat(2).view(batch_size, 2)
                    camera_center = torch.hstack((img_w[:,None], img_h[:,None])) / 2
                
                hmr_output = self.model(
                    batch['img'], bbox_center=batch['bbox_center'], bbox_scale=batch['bbox_scale'], 
                    img_w=batch['img_w'], img_h=batch['img_h'], fl=focal_length,
                )

                pred_pose = hmr_output['pred_pose']
                pred_cam_t = hmr_output['pred_cam_t']
                pred_betas = hmr_output['pred_shape']
                pred_vertices = (
                    hmr_output['vertices'] + hmr_output['pred_cam_t'].unsqueeze(1)
                ).detach().cpu().numpy()
            
                if save_results:
                    imgnames.extend(batch['imgname'])
                    smplx_pose.extend(pred_pose.cpu().numpy())
                    smplx_betas.extend(pred_betas.cpu().numpy())
                    smplx_trans.extend(pred_cam_t.cpu().numpy())
                    cam_focal_l.extend(focal_length.cpu().numpy())
                    smplx_verts.extend(pred_vertices)
                    cam_center.extend(camera_center.cpu().numpy())
                    cam_weak_persp.extend(hmr_output['pred_cam'].cpu().numpy())
                
            tracking_results[person_id].update({
                'smplx_pose': np.array(smplx_pose),
                'smplx_betas': np.array(smplx_betas),
                'smplx_trans': np.array(smplx_trans),
                'cam_focal_l': np.array(cam_focal_l),
                'cam_center': np.array(cam_center),
                'cam_weak_persp': np.array(cam_weak_persp),
                'imgnames': imgnames,
            })
            if render_results:
                tracking_results[person_id]['smplx_verts'] = np.array(smplx_verts)
        
        print(f'Saving BEDLAM-CLIFF results to {output_folder}...')
        joblib.dump(tracking_results, os.path.join(output_folder, 'bedlam_cliff_results.pkl'))
        
        if render_results:
            from ..utils.renderer_pyrd import Renderer
            
            renderer = Renderer(
                focal_length=focal_length[0, 0].item(), img_w=img_w[0].item(), img_h=img_h[0].item(),
                faces=self.smplx_cam_head.smplx.faces,
                same_mesh_color=False
            )
            render_save_dir = os.path.join(output_folder, 'renders')
            os.makedirs(render_save_dir, exist_ok=True)
            
            for k, v in tracking_results.items():

                print(f'Rendering tracklet {k}...')
                for i in tqdm.tqdm(range(v['smplx_verts'].shape[0])):
                    
                    vertices = v['smplx_verts'][i]
                    basename = os.path.basename(v['imgnames'][i])
                    
                    front_view_path = os.path.join(render_save_dir, basename)
                    
                    if os.path.isfile(front_view_path):
                        imgname = front_view_path
                    else:
                        imgname = v['imgnames'][i]
                        
                    img = cv2.cvtColor(cv2.imread(imgname), cv2.COLOR_BGR2RGB)
                    front_view = renderer.render_front_view(vertices[None], bg_img_rgb=img.copy())
                    
                    cv2.imwrite(front_view_path, front_view[:, :, ::-1])
            
            renderer.delete()
            
        return tracking_results
    
    @torch.no_grad()
    def run_on_image_folder(self, all_image_folder, detections, output_folder, visualize_proj=True):
        from ..utils.renderer_pyrd import Renderer
        
        for fold_idx, image_folder in enumerate(all_image_folder):
            image_file_names = [
                os.path.join(image_folder, x)
                for x in os.listdir(image_folder)
                if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg')
            ]
            image_file_names = (sorted(image_file_names))
            for img_idx, img_fname in enumerate(tqdm.tqdm(image_file_names)):
                
                dets = detections[fold_idx][img_idx]
                if len(dets) < 1:
                    continue

                img = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)
                orig_height, orig_width = img.shape[:2]
                inp_images = torch.zeros(len(dets), 3, self.model_cfg.DATASET.IMG_RES,
                                         self.model_cfg.DATASET.IMG_RES, device=self.device, dtype=torch.float)

                batch_size = inp_images.shape[0]
                bbox_scale = []
                bbox_center = []

                for det_idx, det in enumerate(dets):
                    bbox = det
                    bbox_scale.append(bbox[2] / 200.)
                    bbox_center.append([bbox[0], bbox[1]])
                    rgb_img = crop(img, bbox_center[-1], bbox_scale[-1],[self.model_cfg.DATASET.IMG_RES, self.model_cfg.DATASET.IMG_RES])
                    rgb_img = np.transpose(rgb_img.astype('float32'), (2, 0, 1)) / 255.0
                    rgb_img = torch.from_numpy(rgb_img)
                    norm_img = self.normalize_img(rgb_img)
                    inp_images[det_idx] = norm_img.float().to(self.device)

                bbox_center = torch.tensor(bbox_center).cuda().float()
                bbox_scale = torch.tensor(bbox_scale).cuda().float()
                img_h = torch.tensor(orig_height).repeat(batch_size).cuda().float()
                img_w = torch.tensor(orig_width).repeat(batch_size).cuda().float()
                focal_length = ((img_w * img_w + img_h * img_h) ** 0.5).cuda().float()
                hmr_output = self.model(inp_images, bbox_center=bbox_center, bbox_scale=bbox_scale, img_w=img_w, img_h=img_h)

                focal_length = (img_w * img_w + img_h * img_h) ** 0.5
                pred_vertices_array = (hmr_output['vertices'] + hmr_output['pred_cam_t'].unsqueeze(1)).detach().cpu().numpy()
                renderer = Renderer(focal_length=focal_length[0], img_w=img_w[0], img_h=img_h[0],
                                    faces=self.smplx_cam_head.smplx.faces,
                                    same_mesh_color=False)
                front_view = renderer.render_front_view(pred_vertices_array,
                                                        bg_img_rgb=img.copy())
                
                
                pred_pose = hmr_output['pred_pose'].cpu().numpy()
                pred_cam_t = hmr_output['pred_cam_t'].cpu().numpy()
                pred_betas = hmr_output['pred_shape'].cpu().numpy()
                
                result_dict = {
                    'bbox': dets,
                    'smplx_pose': pred_pose,
                    'smplx_betas': pred_betas,
                    'smplx_trans': pred_cam_t,
                    'cam_focal_l': focal_length.cpu().numpy(),
                    'imgnames': img_fname,
                    'img_w': img_w.cpu().numpy(),
                    'img_h': img_h.cpu().numpy(),
                }

                # save rendering results
                basename = img_fname.split('/')[-1]
                basename = basename.replace('.png', '').replace('.jpg', '')
                
                joblib.dump(result_dict, os.path.join(output_folder, f'{basename}.pkl'))
                
                filename = basename + "_pred_%s.jpg" % 'bedlam'
                # filename_orig = basename + "orig_%s.jpg" % 'bedlam'
                front_view_path = os.path.join(output_folder, filename)
                # orig_path = os.path.join(output_folder, filename_orig)
                # logger.info(f'Writing output files to {output_folder}')
                cv2.imwrite(front_view_path, front_view[:, :, ::-1])
                # cv2.imwrite(orig_path, img[:, :, ::-1])
                renderer.delete()

    @torch.no_grad()
    def run_on_hbw_folder(self, all_image_folder, detections, output_folder, data_split='test', visualize_proj=True):
        from ..utils.renderer_pyrd import Renderer
        img_names = []
        verts = []
        image_file_names = []
        for fold_idx, image_folder in enumerate(all_image_folder):
            image_file_names = [
                os.path.join(image_folder, x)
                for x in os.listdir(image_folder)
                if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg')
            ]
            image_file_names = (sorted(image_file_names))
            print(image_folder, len(image_file_names))

            for img_idx, img_fname in tqdm.tqdm(enumerate(image_file_names)):
                if detections:
                    dets = detections[fold_idx][img_idx]
                    if len(dets) < 1:
                        img_names.append('/'.join(img_fname.split('/')[-4:]).replace(data_split + '_small_resolution', data_split))
                        template_verts = self.smplx_cam_head.smplx().vertices[0].detach().cpu().numpy()
                        verts.append(template_verts)
                        continue
                else:
                    match_fname = '/'.join(img_fname.split('/')[-3:])
                    if match_fname not in self.bboxes_dict.keys():
                        img_names.append('/'.join(img_fname.split('/')[-4:]).replace(data_split + '_small_resolution', data_split))
                        template_verts = self.smplx_cam_head.smplx().vertices[0].detach().cpu().numpy()
                        verts.append(template_verts)
                        continue
                    dets = self.bboxes_dict[match_fname]
                img = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)
                orig_height, orig_width = img.shape[:2]
                inp_images = torch.zeros(1, 3, self.model_cfg.DATASET.IMG_RES,
                                         self.model_cfg.DATASET.IMG_RES, device=self.device, dtype=torch.float)
                batch_size = inp_images.shape[0]
                bbox_scale = []
                bbox_center = []
                if len(dets.shape)==1:
                    dets = np.expand_dims(dets, 0)
                for det_idx, det in enumerate(dets):
                    if det_idx>=1:
                        break
                    bbox = det
                    bbox_scale.append(bbox[2] / 200.)
                    bbox_center.append([bbox[0], bbox[1]])
                    rgb_img = crop(img, bbox_center[-1], bbox_scale[-1],[self.model_cfg.DATASET.IMG_RES, self.model_cfg.DATASET.IMG_RES])
                    rgb_img = np.transpose(rgb_img.astype('float32'), (2, 0, 1)) / 255.0
                    rgb_img = torch.from_numpy(rgb_img)
                    norm_img = self.normalize_img(rgb_img)
                    inp_images[det_idx] = norm_img.float().to(self.device)

                bbox_center = torch.tensor(bbox_center).cuda().float()
                bbox_scale = torch.tensor(bbox_scale).cuda().float()
                img_h = torch.tensor(orig_height).repeat(batch_size).cuda().float()
                img_w = torch.tensor(orig_width).repeat(batch_size).cuda().float()
                focal_length = ((img_w * img_w + img_h * img_h) ** 0.5).cuda().float()

                hmr_output = self.model(inp_images, bbox_center=bbox_center, bbox_scale=bbox_scale, img_w=img_w, img_h=img_h)
                img_names.append('/'.join(img_fname.split('/')[-4:]).replace(data_split + '_small_resolution', data_split))
                template_verts = self.smplx_cam_head.smplx(betas=hmr_output['pred_shape'], pose2rot=False).vertices[0].detach().cpu().numpy()
                verts.append(template_verts)
                if visualize_proj:
                    focal_length = (img_w * img_w + img_h * img_h) ** 0.5

                    pred_vertices_array = (hmr_output['vertices'][0] + hmr_output['pred_cam_t']).unsqueeze(0).detach().cpu().numpy()
                    renderer = Renderer(focal_length=focal_length, img_w=img_w, img_h=img_h,
                                        faces=self.smplx_cam_head.smplx.faces,
                                        same_mesh_color=False)
                    front_view = renderer.render_front_view(pred_vertices_array,
                                                            bg_img_rgb=img.copy())

                    # save rendering results
                    basename = img_fname.split('/')[-3]+'_'+img_fname.split('/')[-2]+'_'+img_fname.split('/')[-1]
                    filename = basename + "pred_%s.jpg" % 'bedlam'
                    filename_orig = basename + "orig_%s.jpg" % 'bedlam'
                    front_view_path = os.path.join(output_folder, filename)
                    orig_path = os.path.join(output_folder, filename_orig)
                    logger.info(f'Writing output files to {output_folder}')
                    cv2.imwrite(front_view_path, front_view[:, :, ::-1])
                    cv2.imwrite(orig_path, img[:, :, ::-1])
                    renderer.delete()
        np.savez(os.path.join(output_folder, data_split + '_hbw_prediction.npz'), image_name=img_names, v_shaped=verts)
               
    def run_on_dataframe(self, dataframe_path, output_folder, visualize_proj=True):
        from ..utils.renderer_pyrd import Renderer
        dataframe = np.load(dataframe_path)
        centers = dataframe['center']
        scales = dataframe['scale']
        image = dataframe['image']
        for ind, center in tqdm.tqdm(enumerate(centers)):
            center = centers[ind]
            scale = scales[ind]
            img = image[ind]
            orig_height, orig_width = img.shape[:2]
            rgb_img = crop(img, center, scale, [self.hparams.DATASET.IMG_RES, self.hparams.DATASET.IMG_RES])

            rgb_img = np.transpose(rgb_img.astype('float32'), (2, 0, 1)) / 255.0
            rgb_img = torch.from_numpy(rgb_img).float().cuda()
            rgb_img = self.normalize_img(rgb_img)

            img_h = torch.tensor(orig_height).repeat(1).cuda().float()
            img_w = torch.tensor(orig_width).repeat(1).cuda().float()
            center = torch.tensor(center).cuda().float()
            scale = torch.tensor(scale).cuda().float()

            hmr_output = self.model(rgb_img.unsqueeze(0), bbox_center=center.unsqueeze(0), bbox_scale=scale.unsqueeze(0), img_w=img_w, img_h=img_h)
            # Need to convert SMPL-X meshes to SMPL using conversion tool before calculating error
            import trimesh
            mesh = trimesh.Trimesh(vertices=hmr_output['vertices'][0].detach().cpu().numpy(),faces=self.smplx_cam_head.smplx.faces)
            output_mesh_path = os.path.join(output_folder, str(ind)+'.obj')
            mesh.export(output_mesh_path)

            if visualize_proj:
                focal_length = (img_w * img_w + img_h * img_h) ** 0.5

                pred_vertices_array = (hmr_output['vertices'][0] + hmr_output['pred_cam_t']).unsqueeze(0).detach().cpu().numpy()
                renderer = Renderer(focal_length=focal_length, img_w=img_w, img_h=img_h,
                                    faces=self.smplx_cam_head.smplx.faces,
                                    same_mesh_color=False)
                front_view = renderer.render_front_view(pred_vertices_array,
                                                        bg_img_rgb=img.copy())

                # save rendering results
                basename = str(ind)
                filename = basename + "pred_%s.jpg" % 'bedlam'
                filename_orig = basename + "orig_%s.jpg" % 'bedlam'
                front_view_path = os.path.join(output_folder, filename)
                orig_path = os.path.join(output_folder, filename_orig)
                logger.info(f'Writing output files to {output_folder}')
                cv2.imwrite(front_view_path, front_view[:, :, ::-1])
                cv2.imwrite(orig_path, img[:, :, ::-1])
                renderer.delete()