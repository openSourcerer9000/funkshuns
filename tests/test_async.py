import pytest
import aiofiles
from aiofiles.os import remove as async_remove
import pandas as pd
from unittest.mock import MagicMock
from io import StringIO
from funkshuns.funkshuns import *

mc = Path(os.path.dirname(os.path.abspath(__file__)))/'mockdata'/'async'

content_txt = 'column1,column2\n1,2\n3,4\n'

# @pytest.mark.asyncio
@pytest.fixture
def csv_file():
    test_file_path = mc/'test.csv'
    
    test_file_path.write_text(content_txt)

    # async with aiofiles.open(test_file_path, mode='w') as f:
    #     await f.write(content_txt)
    yield test_file_path  

    test_file_path.unlink(missing_ok=True)
    # await async_remove(test_file_path)  # cleanup after test function finishes

@pytest.mark.asyncio
async def test__read_csv_async(csv_file):
    print('csvvv',csv_file,type(csv_file))
    content = await xpd._read_csv_async(csv_file)
    expect_content = 'column1,column2\n1,2\n3,4\n'
    assert content == expect_content, f"Expected: {expect_content}, but got: {content}"

@pytest.mark.asyncio
async def test_read_csv_async(csv_file):
    df = await xpd.read_csv_async(csv_file)
    expect_df = pd.read_csv(StringIO(content_txt))

    assert df.equals(expect_df), "DataFrame object is not as expected."

@pytest.mark.asyncio
async def test_concurrent_read_csv_async(csv_file):
    # Call read_csv_async multiple times at once
    tasks = [asyncio.create_task(xpd.read_csv_async(csv_file)) for _ in range(50)]
    dfs = await asyncio.gather(*tasks)

    expect_df = pd.read_csv(StringIO(content_txt))

    # Check that each DataFrame is equivalent to the expected DataFrame
    for df in dfs:
        assert df.equals(expect_df), "DataFrame object is not as expected."

# Additional test case to verify whether xpd._read_csv_async() is called in xpd.read_csv_async()
# @pytest.mark.asyncio
# async def test_read_csv_async_integration():
#     test_csvpth = "test_file"
#     test_kwargz = {"delimiter": ";"}

#     mock_content = "mock_content"
#     mock_df = pd.DataFrame()

#     # Mock object for xpd._read_csv_async
#     mock__read_csv_async = MagicMock()
#     mock__read_csv_async.return_value = mock_content
#     xpd._read_csv_async_origin = xpd._read_csv_async
#     xpd._read_csv_async = mock__read_csv_async

#     # Mock object for pd.read_csv
#     mock_read_csv = MagicMock()
#     mock_read_csv.return_value = mock_df
#     pd.read_csv_origin = pd.read_csv
#     pd.read_csv = mock_read_csv

#     df = await xpd.read_csv_async(test_csvpth, **test_kwargz)

#     mock__read_csv_async.assert_called_once_with(test_csvpth)
#     mock_read_csv.assert_called_once_with(StringIO(mock_content), **test_kwargz)

#     assert df.equals(mock_df), "DataFrame object is not as expected."

#     # Reset the mocked functions
#     xpd._read_csv_async = xpd._read_csv_async_origin    
#     pd.read_csv = pd.read_csv_origin