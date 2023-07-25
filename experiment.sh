#! /bin/sh
search_path="./../choiDataset/"
for file_path_lv0 in $search_path*
do
	if [ -d $file_path_lv0 ]; then
#		echo $file_path_lv0
		if [ ! -d "${file_path_lv0/"choiDataset"/choiResult4kYESYEStshort}" ]; then
			mkdir "${file_path_lv0/"choiDataset"/choiResult4kYESYEStshort}"
		fi
		for file_path_lv1 in "$file_path_lv0"/*
		do
			if [ -d $file_path_lv1 ]; then
				if [ ! -d "${file_path_lv1/"choiDataset"/choiResult4kYESYEStshort}" ]; then
					mkdir "${file_path_lv1/"choiDataset"/choiResult4kYESYEStshort}"
				fi
#				python3 workflow_segmentation.py --dir_path $file_path_lv1 --gpt_model_type gpt-3.5-turbo-16k --prevent_long_segment_flag ON &
#				python3 workflow_segmentation.py --dir_path $file_path_lv1 --gpt_model_type gpt-3.5-turbo-16k --prevent_long_segment_flag OFF &
#				python3 workflow_segmentation.py --dir_path $file_path_lv1 --gpt_model_type gpt-3.5-turbo --prevent_long_segment_flag ON &
#				python3 workflow_segmentation.py --dir_path $file_path_lv1 --gpt_model_type gpt-3.5-turbo --prevent_long_segment_flag OFF &
#				python3 workflow_segmentation.py --dir_path $file_path_lv1 --gpt_model_type gpt-4 --prevent_long_segment_flag OFF &
#				python3 workflow_segmentation.py --dir_path $file_path_lv1 --gpt_model_type gpt-4 --prevent_long_segment_flag ON &

#				python3 workflow_segmentation_short_response.py --dir_path $file_path_lv1 --gpt_model_type gpt-3.5-turbo-16k --prevent_long_segment_flag ON &
#				python3 workflow_segmentation_short_response.py --dir_path $file_path_lv1 --gpt_model_type gpt-3.5-turbo-16k --prevent_long_segment_flag OFF &
#				python3 workflow_segmentation_short_response.py --dir_path $file_path_lv1 --gpt_model_type gpt-3.5-turbo --prevent_long_segment_flag ON &
#				python3 workflow_segmentation_short_response.py --dir_path $file_path_lv1 --gpt_model_type gpt-3.5-turbo --prevent_long_segment_flag OFF &
#				python3 workflow_segmentation_short_response.py --dir_path $file_path_lv1 --gpt_model_type gpt-4 --prevent_long_segment_flag ON &
#				python3 workflow_segmentation_short_response.py --dir_path $file_path_lv1 --gpt_model_type gpt-4 --prevent_long_segment_flag OFF &

				python3 workflow_segmentation_short_response.py --dir_path $file_path_lv1 --gpt_model_type gpt-3.5-turbo-16k --prevent_long_segment YES --prevent_short_segment YES &
				python3 workflow_segmentation_short_response.py --dir_path $file_path_lv1 --gpt_model_type gpt-3.5-turbo-16k --prevent_long_segment NO --prevent_short_segment YES &
				python3 workflow_segmentation_short_response.py --dir_path $file_path_lv1 --gpt_model_type gpt-3.5-turbo --prevent_long_segment YES --prevent_short_segment YES &
				python3 workflow_segmentation_short_response.py --dir_path $file_path_lv1 --gpt_model_type gpt-3.5-turbo --prevent_long_segment NO --prevent_short_segment YES &
			fi
		done
	fi
done
