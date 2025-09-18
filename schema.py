from pymilvus import DataType, MilvusClient

faceSchema = MilvusClient.create_schema()

faceSchema.add_field(
    field_name="id",
    datatype=DataType.INT64,
    is_primary=True,
    auto_id=True,
)

faceSchema.add_field(field_name="code", datatype=DataType.VARCHAR)
faceSchema.add_field(field_name="pose", datatype=DataType.INT8)
faceSchema.add_field(field_name="pose", datatype=DataType.FLOAT_VECTOR, dim=512)
