#! /bin/sh

python3 gen_SOP.py --gpt_model_type gpt-3.5-turbo-16k --prevent_long_segment YES --prevent_short_segment YES --file_path ./../transcription/transWellington.txt
