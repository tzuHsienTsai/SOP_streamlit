import logging
import os
import time
import utils
from flask import Flask, jsonify, request
import warnings

import CloudRunService
from config import configure_logging
from deephow.internal.admin.database.database_service import DatabaseService
from deephow.internal.model_data.capture_model_data import CaptureModelData
from segmentation import segmentation_entry


SERVICE = "segmentation"
VERSION = "2.2"

# service init
warnings.filterwarnings("ignore")
configure_logging()
app = Flask(__name__)
db = DatabaseService().instance()
_, model_data_config = db.get_document('configurations', 'modelData')
model_data = CaptureModelData(model_data_config, logging)


@app.errorhandler(Exception)
def handle_exception(e):
    error_msg = str(e)
    logging.error(error_msg)
    return jsonify(f'Server side error: {error_msg}'), 500


@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'return_responses': 'health check pass !'}), 200


@app.route('/', methods=['POST'])
def main():
    request_data = request.get_json()
    try:
        utils.validate_input_schema(request_data)
    except Exception as e:
        logging.error(f'request error: {e}')
        result = {
            "message": "Call segmentation error",
            "workflowId": request_data.get('workflowId', '')
        }
        if 'workflowId' in request_data:
            logging.info('callback to step-server: Call segmentation error')
            try:
                utils.callback(result, 1)
            except Exception:
                result["message"] = "Call segmentation error, and callback error"

        return jsonify(result), 400

    start_time = time.time()
    workflow_id = request_data['workflowId']
    lang = request_data['lang']
    logging.info(f'process starting...{len(request_data["input"])}')
    CloudRunService.total_response_size = 0
    try:
        result = segmentation_entry(request_data)
        try:
            # output capture
            if not request_data.get('isSlides'):
                end_time = time.time()
                _, workflow_doc = db.get_document('workflows', workflow_id)
                model_data.save_segmentation(
                    workflow_id,
                    workflow_doc.get('videoDuration'),
                    lang,
                    request_data['input'],
                    result['segmented_text'],
                    result['summaries'],
                    result['video_tagging'],
                    SERVICE,
                    VERSION,
                    workflow_doc.get('organization'),
                    end_time - start_time
                )
        except Exception as e:
            logging.exception(f"save model data error: {e}")

        try:
            utils.callback(result, 1)
        except Exception:
            result['message'] = 'Call segmentation success, but callback error'
            return jsonify(result), 400
        return jsonify(result), 200

    except Exception as e:
        logging.error(f'Server side error: {e}', exc_info=True)
        logging.info('callback to step-server: Call segmentation error')
        result = {"message": "Call segmentation error", "workflowId": workflow_id}
        try:
            utils.callback(result, 1)
        except Exception:
            result["message"] = "Call segmentation error, and callback error"
        return jsonify(result), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv('PORT', 8080)))
