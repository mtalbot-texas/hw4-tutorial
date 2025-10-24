Get the key

Open this page in a browser and sign in with your Google account
https://aistudio.google.com/app/apikey

Click Create API key. You'll need to use your personal GMAIL as utexas blocks usage of this key

Copy the key and keep it private.

python -m streamlit run app.py

Navigate to:
https://console.cloud.google.com/bigquery
Install Google CLI:
# Download and install to $HOME/google-cloud-sdk
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-darwin-x86_64.tar.gz
tar -xzf google-cloud-cli-darwin-x86_64.tar.gz
./google-cloud-sdk/install.sh

Run to Authenticate with the account you used with your prior google big query project
gcloud auth application-default login
gcloud config set project indigo-proxy-472718-m1
