create procedure load_data as
begin

truncate table dbo.dynamic_pricing
bulk insert dbo.dynamic_pricing
from 'C:\SQL_PROJECTS\End-to-End-Real-Time-Dynamic-Pricing-Engine\sql\raw_data\dynamic_pricing.csv'
with
(
firstrow = 2,
fieldterminator = ',',
tablock
)
end
