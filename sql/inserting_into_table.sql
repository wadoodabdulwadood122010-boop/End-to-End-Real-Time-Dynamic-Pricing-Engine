CREATE OR ALTER PROCEDURE load_data AS
BEGIN
    -- 1. Clean out the target table
    TRUNCATE TABLE dbo.dynamic_pricing;

    -- 2. Create a temporary staging table with NO identity column
    -- Ensure the columns match the exact sequence of your CSV file
    IF OBJECT_ID('tempdb..#Staging_Pricing') IS NOT NULL 
        DROP TABLE #Staging_Pricing;

    CREATE TABLE #Staging_Pricing (
        Number_of_Riders INT,
        Number_of_Drivers INT,
        Location_Category VARCHAR(50),
        Customer_Loyalty_Status VARCHAR(50),
        Number_of_Past_Rides INT,
        Average_Ratings FLOAT,
        Time_of_Booking VARCHAR(50),
        Vehicle_Type VARCHAR(50),
        Expected_Ride_Duration INT,
        Historical_Cost_of_Ride FLOAT
    );

    -- 3. Bulk insert directly into the staging table
    BULK INSERT #Staging_Pricing
    FROM 'C:\SQL_PROJECTS\End-to-End-Real-Time-Dynamic-Pricing-Engine\sql\raw_data\dynamic_pricing.csv'
    WITH
    (
        FORMAT = 'CSV',
        FIRSTROW = 2,
        FIELDTERMINATOR = ',',
        ROWTERMINATOR = '\n',
        TABLOCK
    );

    -- 4. Move data securely into your production table
    -- Explicitly name the columns, skipping your primary key column completely
    INSERT INTO dbo.dynamic_pricing (
        Number_of_request, -- Use whatever your final table column names are
        Number_of_Drivers,
        Location_Category,
        Customer_Loyalty_Status,
        Number_of_Past_Rides,
        Average_rating,
        Time_of_boking,
        Vehicle_Type,
        Expected_Ride_Duration,
        Historical_Cost_of_Ride
    )
    SELECT 
        Number_of_Riders,
        Number_of_Drivers,
        Location_Category,
        Customer_Loyalty_Status,
        Number_of_Past_Rides,
        Average_Ratings,
        Time_of_Booking,
        Vehicle_Type,
        Expected_Ride_Duration,
        Historical_Cost_of_Ride
    FROM #Staging_Pricing;

    -- Clean up the temporary table
    DROP TABLE #Staging_Pricing;
END;
GO

EXEC load_data;