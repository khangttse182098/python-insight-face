from pymilvus import DataType, MilvusClient

face_schema = MilvusClient.create_schema()

face_schema.add_field(
    field_name="id",
    datatype=DataType.INT64,
    is_primary=True,
    auto_id=True,
)
face_schema.add_field(field_name="code", datatype=DataType.VARCHAR, max_length=10)
face_schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=512)
