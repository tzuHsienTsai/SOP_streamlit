#! /bin/sh
#python3 count_length.py --gpt_model_type gpt-3.5-turbo-16k --prevent_long_segment YES --prevent_short_segment YES
#python3 count_length.py --gpt_model_type gpt-3.5-turbo-16k --prevent_long_segment YES --prevent_short_segment NO
python3 count_length.py --gpt_model_type gpt-3.5-turbo-16k --prevent_long_segment NO --prevent_short_segment YES
python3 count_length.py --gpt_model_type gpt-3.5-turbo-16k --prevent_long_segment NO --prevent_short_segment NO
