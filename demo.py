import os
import sys
import argparse
import joblib
from loguru import logger
from glob import glob

import numpy as np
from train.core.tester import Tester

# os.environ['PYOPENGL_PLATFORM'] = 'egl'
sys.path.append('')


def main(args):

    input_image_folder = args.image_folder
    output_path = args.output_folder
    os.makedirs(output_path, exist_ok=True)

    logger.add(
        os.path.join(output_path, 'demo.log'),
        level='INFO',
        colorize=False,
    )
    logger.info(f'Demo options: \n {args}')

    tester = Tester(args)
    if args.eval_dataset == 'hbw':
        all_image_folder = glob(os.path.join(input_image_folder, 'images', args.data_split + '_small_resolution', '*', '*'))
        all_bbox_folder = glob(os.path.join('data/test_images/hbw_test_images_bbox', '*', '*','labels'))
        tester.load_yolov5_bboxes(all_bbox_folder)
        detections = None
        tester.run_on_hbw_folder(all_image_folder, detections, output_path, args.data_split, args.display)
    elif args.eval_dataset == 'ssp':
        dataframe_path = args.dataframe_path
        tester.run_on_dataframe(dataframe_path, output_path, args.display)
    elif args.eval_dataset == 'video':
        
        mpt_res_path = os.path.join(output_path, 'mpt_results.pkl')
        if args.bbox_file is not None:
            bboxes = np.loadtxt(args.bbox_file)
            tracking_results = {
                1: {
                    'bbox': bboxes,
                    'frames': np.arange(len(bboxes)),
                },
            }
            joblib.dump(tracking_results, mpt_res_path)
        else:
            if os.path.exists(mpt_res_path):
                logger.info('Loading tracking results from cache', mpt_res_path)
                tracking_results = joblib.load(mpt_res_path)
            else:
                tracking_results = tester.run_tracking(input_image_folder)
                joblib.dump(tracking_results, os.path.join(output_path, 'mpt_results.pkl'))
                
        tester.run_on_video(
            input_image_folder, 
            tracking_results, 
            args.output_folder, 
            render_results=args.enable_render,
            cam_intrinsics_file=args.cam_intrinsics
        )
    else:
        all_image_folder = [input_image_folder]
        detections = tester.run_detector(all_image_folder)
        tester.run_on_image_folder(all_image_folder, detections, output_path, args.display, render_results=args.enable_render)

    del tester.model

    logger.info('================= END =================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, default='configs/demo_bedlam_cliff.yaml',
                        help='config file that defines model hyperparams')

    parser.add_argument('--ckpt', type=str, default='data/ckpt/bedlam_cliff.ckpt',
                        help='checkpoint path')

    parser.add_argument('--image_folder', type=str, default='demo_images',
                        help='input image folder')

    parser.add_argument('--output_folder', type=str, default='demo_images/results',
                        help='output folder to write results')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size for inference')
    
    parser.add_argument('--tracker_batch_size', type=int, default=1,
                        help='batch size of object detector used for bbox tracking')
    
    parser.add_argument('--cam_intrinsics', type=str, default=None, help='camera intrinsics file')
    
    parser.add_argument('--bbox_file', type=str, default=None, help='bbox file')
                        
    parser.add_argument('--display', action='store_true',
                        help='visualize the 3d body projection on image')

    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'],
                        help='object detector to be used for bbox tracking')

    parser.add_argument('--yolo_img_size', type=int, default=416,
                        help='input image size for yolo detector')
    parser.add_argument('--eval_dataset', type=str, default=None)
    parser.add_argument('--dataframe_path', type=str, default='data/ssp_3d_test.npz')
    parser.add_argument('--data_split', type=str, default='test')
    parser.add_argument('--enable_render', action='store_true')

    args = parser.parse_args()
    main(args)
