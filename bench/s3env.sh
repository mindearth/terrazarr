# Source this to export MinIO credentials from ~/repos/.myenvs for obstore, s3fs and GDAL.
# usage: source bench/s3env.sh
set -a; . "${MYENVS:-$HOME/repos/.myenvs}"; set +a
export AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY"
export AWS_ENDPOINT_URL="$AWS_ENDPOINT_URL"
export AWS_REGION="us-east-1"
# GDAL /vsis3/
export AWS_S3_ENDPOINT="$AWS_ENDPOINT_URL"
export AWS_HTTPS="NO"
export AWS_VIRTUAL_HOSTING="FALSE"
unset AWS_PROFILE AWS_DEFAULT_PROFILE
# botocore >= 1.36 stops sending Content-MD5 by default; MinIO requires it for DeleteObjects (s3fs bulk deletes)
export AWS_REQUEST_CHECKSUM_CALCULATION="when_required"
export AWS_RESPONSE_CHECKSUM_VALIDATION="when_required"
