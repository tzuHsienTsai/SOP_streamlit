export PORT=8080
export LOGLEVEL=INFO
export OPENAI_API_KEY=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
export configuration_file_path=config.json

# dev
# export u_s_e_address=https://universal-sentence-encoder-3oqy6iorxq-uc.a.run.app
# export temporal_embeddings_gen_address=https://temporal-embeddings-gen-3oqy6iorxq-uc.a.run.app
# export mt5_address=https://mt5-sentence-encoder-3oqy6iorxq-uc.a.run.app
# export vit_address=https://vit-embeddings-gen-3oqy6iorxq-uc.a.run.app
# export entities_and_tags_gen_address=https://entities-and-tags-3oqy6iorxq-uc.a.run.app
# export ernie_address=https://ernie-3oqy6iorxq-uc.a.run.app
export GOOGLE_APPLICATION_CREDENTIALS=../../../deephow-dev-d662831af27f.json
export project_id=deephow-dev

# prod
# export GOOGLE_APPLICATION_CREDENTIALS=/home/yunda_tsai_deephow_com/.config/gcloud/application_default_credentials.json
# export mt5_address=https://mt5-sentence-encoder-kfk24stg4a-uc.a.run.app
# export vit_address=https://vit-embeddings-gen-kfk24stg4a-uc.a.run.app
# export callback_url=https://steps-dot-deephow-prod.uc.r.appspot.com/steps/segmentation-callback
# export project_id=deephow-prod

export GPU=True

exec gunicorn --bind :$PORT --workers 2 --timeout 900 main:app
