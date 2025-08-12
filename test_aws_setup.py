import boto3
import sagemaker
import os
from dotenv import load_dotenv

load_dotenv()

BUCKET = os.getenv('BUCKET')
ROLE_ARN = os.getenv('ROLE_ARN')

def test_complete_setup():
    try:
        # Test SageMaker session
        sess = sagemaker.Session()
        print(f"✅ SageMaker session: {sess.boto_region_name}")
        
        # Test S3 bucket
        s3 = boto3.client('s3')
        try:
            s3.head_bucket(Bucket=BUCKET)
            print(f"✅ S3 bucket accessible: {BUCKET}")
        except Exception as e:
            print(f"❌ S3 bucket not accessible: {e}")
            return False
        
        # Test role
        iam = boto3.client('iam')
        try:
            role = iam.get_role(RoleName="SageMakerExecutionRole-Pipeline-MXNet")
            print(f"✅ Role accessible: {role['Role']['RoleName']}")
        except Exception as e:
            print(f"❌ Role not accessible: {e}")
            return False
        
        print(f"\n🚀 EVERYTHING IS READY!")
        print(f"Role ARN: {ROLE_ARN}")
        print(f"Bucket: {BUCKET}")
        
        return True
        
    except Exception as e:
        print(f"❌ Issue: {e}")
        return False

if __name__ == "__main__":
    test_complete_setup()



# python sagemaker_main.py train \
#     --bucket plastic-bag-mxnet-503561429929 \
#     --prefix plastic-bags \
#     --epochs 10 \
#     --instance-type ml.m4.xlarge