import asyncio
import aioboto3


async def test_connect(key_id, access_key, bucket):
    s3_endpoint_url = "https://obs.ru-moscow-1.hc.sbercloud.ru"
    session = aioboto3.Session()
    client = session.client(
        service_name="s3",
        endpoint_url=s3_endpoint_url,
        aws_access_key_id=key_id,
        aws_secret_access_key=access_key,
        use_ssl=False,
        verify=False,
    )
    async with client as s3:
        Key = "test_key"
        await s3.put_object(Bucket=bucket, Key=Key, Body=b"1,2,4")
        await s3.get_object(Bucket=bucket, Key=Key)
        await s3.delete_object(Bucket=bucket, Key=Key)
        print(f"Ok for {bucket}")


async def upload(
                filename: str,
                staging_path,
                key_id: str, 
                access_key: str, 
                bucket: str,
                ) -> str:
                
    blob_s3_key = f"{filename}"

    s3_endpoint_url = "https://obs.ru-moscow-1.hc.sbercloud.ru"
    session = aioboto3.Session()
    client = session.client(
        service_name="s3",
        endpoint_url=s3_endpoint_url,
        aws_access_key_id=key_id,
        aws_secret_access_key=access_key,
        use_ssl=False,
        verify=False,
    )

    async with client as s3:
        try:
            #with open(staging_path) as spfp:
            print(f"Uploading {blob_s3_key} to s3")
            Key = staging_path
            await s3.put_object(Bucket=bucket, Key=Key, Body=staging_path)
            print(f"Finished Uploading {blob_s3_key} to s3")
        except Exception as e:
            print(f"Unable to s3 upload {staging_path} to {blob_s3_key}: {e} ({type(e)})")
            return ""

    return f"s3://{blob_s3_key}"


if __name__ == "__main__":
    asyncio.run(
        test_connect(
            key_id="ISI2088GEMLH8RACBTXC",
            access_key="NWzeBtZdHHLIlwaaSOVUIkh3qukzhc5japhrdXwx",
            bucket="hackathon-ecs-31",
        )
    )

    filename = "test_video.json"
    path = "../test.json"
    
    #filename = "vid_test_result.mp4"
    #path = "../hackathon_part_1.mp4"

    asyncio.run(
        upload(
            filename=filename,
            staging_path=path,
            key_id="ISI2088GEMLH8RACBTXC",
            access_key="NWzeBtZdHHLIlwaaSOVUIkh3qukzhc5japhrdXwx",
            bucket="hackathon-ecs-31",
        )
    )