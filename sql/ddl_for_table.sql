if object_id('[dynamic_pricing]', 'U') is not null
	drop table [dynamic_pricing]
create table [dynamic_pricing]
(
primary_key INT IDENTITY(1,1) NOT NULL,
Number_of_request int,
Number_of_Drivers int,
Location_Category varchar(50),
Customer_Loyalty_Status varchar(50),  
Number_of_Past_Rides int,
Average_rating float,
Time_of_boking  varchar(50),
Vehicle_Type varchar(50),
Expected_Ride_Duration int,
Historical_Cost_of_Ride float
)