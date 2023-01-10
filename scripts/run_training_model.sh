cd src/train/data && curl https://s3-ap-southeast-1.amazonaws.com/kobiton-devvn/downloads/AI/sample_data.zip -o sample_data.zip && unzip sample_data.zip && cd ../../
python -m train.mutimodal.tftrain

# Copy trained model files to deploy in the AI service
cp -r train/models/saved_model/* service/models/1