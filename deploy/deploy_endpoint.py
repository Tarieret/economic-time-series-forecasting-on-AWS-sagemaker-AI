import sagemaker
from sagemaker import get_execution_role
from sagemaker.xgboost.model import XGBoostModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

session = sagemaker.Session()
role = get_execution_role()
s3_path = session.upload_data("xgboost-model.tar.gz",
                              bucket=session.default_bucket(),
                              key_prefix="cpi-forecaster-xgboost")

model = XGBoostModel(model_data=s3_path, role=role,
                     entry_point="inference.py",
                     framework_version="3.0-5")
predictor = model.deploy(initial_instance_count=1,
                         instance_type="ml.m5.large",
                         serializer=JSONSerializer(),
                         deserializer=JSONDeserializer())
print("Endpoint:", predictor.endpoint_name)
try:
    print(predictor.predict({"last_value": 325.0, "recent_changes": [0.3] * 12}))
finally:
    predictor.delete_endpoint()
    print("Endpoint deleted")
