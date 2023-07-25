#! /bin/sh
search_path="./../choiDataset/"
#echo "===gpt3.5 16k YES NO==="
#python3 workflow_segmentation.py --dir_path $file_path_lv1 --gpt_model_type gpt-3.5-turbo-16k --prevent_long_segment YES --prevent_short_segment NO --file_name Bike
#echo "===gpt3.5 16k NO NO==="
#python3 workflow_segmentation.py --dir_path $file_path_lv1 --gpt_model_type gpt-3.5-turbo-16k --prevent_long_segment NO --prevent_short_segment NO --file_name Bike
#echo "===gpt3.5 4k YES NO==="
#python3 workflow_segmentation.py --dir_path $file_path_lv1 --gpt_model_type gpt-3.5-turbo --prevent_long_segment YES --prevent_short_segment NO --file_name Bike
#echo "===gpt3.5 4k NO NO==="
#python3 workflow_segmentation.py --dir_path $file_path_lv1 --gpt_model_type gpt-3.5-turbo --prevent_long_segment NO --prevent_short_segment NO --file_name Bike

#echo "===gpt3.5 4k YES NO short==="
#python3 workflow_segmentation_short_response.py --dir_path $file_path_lv1 --gpt_model_type gpt-3.5-turbo --prevent_long_segment YES --prevent_short_segment NO --file_name Bike
#echo "===gpt3.5 4k NO NO short==="
#python3 workflow_segmentation_short_response.py --dir_path $file_path_lv1 --gpt_model_type gpt-3.5-turbo --prevent_long_segment NO --prevent_short_segment NO --file_name Bike
#echo "===gpt3.5 16k YES NO short==="
#python3 workflow_segmentation_short_response.py --dir_path $file_path_lv1 --gpt_model_type gpt-3.5-turbo-16k --prevent_long_segment YES --prevent_short_segment NO --file_name Bike
#echo "===gpt3.5 16k NO NO short==="
#python3 workflow_segmentation_short_response.py --dir_path $file_path_lv1 --gpt_model_type gpt-3.5-turbo-16k --prevent_long_segment NO --prevent_short_segment NO --file_name Bike
echo "===gpt3.5 4k YES YES short==="
python3 workflow_segmentation_short_response.py --dir_path $file_path_lv1 --gpt_model_type gpt-3.5-turbo --prevent_long_segment YES --prevent_short_segment YES --file_name Bike
#echo "===gpt3.5 4k NO YES short==="
#python3 workflow_segmentation_short_response.py --dir_path $file_path_lv1 --gpt_model_type gpt-3.5-turbo --prevent_long_segment NO --prevent_short_segment YES --file_name Bike
echo "===gpt3.5 16k YES YES short==="
python3 workflow_segmentation_short_response.py --dir_path $file_path_lv1 --gpt_model_type gpt-3.5-turbo-16k --prevent_long_segment YES --prevent_short_segment YES --file_name Bike
#echo "===gpt3.5 16k NO YES short==="
#python3 workflow_segmentation_short_response.py --dir_path $file_path_lv1 --gpt_model_type gpt-3.5-turbo-16k --prevent_long_segment NO --prevent_short_segment YES --file_name Bike

#echo "===gpt4 8k YES==="
#python3 workflow_segmentation.py --dir_path $file_path_lv1 --gpt_model_type gpt-4 --prevent_long_segment YES --file_name Bike
#echo "===gpt4 8k NO==="
#python3 workflow_segmentation.py --dir_path $file_path_lv1 --gpt_model_type gpt-4 --prevent_long_segment NO --file_name Bike
#echo "===gpt4 8k YES short==="
#python3 workflow_segmentation_short_response.py --dir_path $file_path_lv1 --gpt_model_type gpt-4 --prevent_long_segment YES --file_name Bike
#echo "===gpt4 8k NO short==="
#python3 workflow_segmentation_short_response.py --dir_path $file_path_lv1 --gpt_model_type gpt-4 --prevent_long_segment NO --file_name Bike
