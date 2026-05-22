
-- 1. Dimension exploration

select distinct location_Category
FROM dynamic_pricing


select distinct Customer_Loyalty_Status
FROM dynamic_pricing


select distinct Time_of_boking
FROM dynamic_pricing


select distinct Vehicle_Type
FROM dynamic_pricing



--2. Measure Exploration 

select 
avg(Number_of_request) avg_number_of_requests,
avg(Number_of_Drivers) avg_Number_of_Drivers,
avg(Expected_Ride_Duration) avg_Expected_Ride_Duration,
avg(Historical_Cost_of_Ride) avg_cost
from dynamic_pricing


-- 3. Magnitude analysis

-- Total (requests and drivers) by location
select Location_Category, sum(Number_of_request) Number_of_request, sum(Number_of_Drivers) Number_of_Drivers 
from dynamic_pricing
group by Location_Category
order by Number_of_request desc

-- Total (requests and drivers) by time of booking
select Time_of_boking, sum(Number_of_request) Number_of_request, sum(Number_of_Drivers) Number_of_Drivers 
from dynamic_pricing
group by Time_of_boking
order by Number_of_request desc

-- Total (requests and drivers) by Vehicle_type
select Vehicle_Type, sum(Number_of_request) Number_of_request, sum(Number_of_Drivers) Number_of_Drivers 
from dynamic_pricing
group by Vehicle_Type
order by Number_of_request desc



-- 4. Part-to-whole analysis


-- Time_of_booking contribution in Total cost
select Time_of_boking, sum(Historical_Cost_of_Ride) costs_of_catgory,
cast(sum(Historical_Cost_of_Ride) as decimal) / sum(sum(Historical_Cost_of_Ride)) over() * 100 as [contribution %]
from dynamic_pricing
group by Time_of_boking
order by cast(sum(Historical_Cost_of_Ride) as decimal) / sum(sum(Historical_Cost_of_Ride)) over() * 100 desc


-- location_Category contribution in Total cost
select location_Category, sum(Historical_Cost_of_Ride) costs_of_catgory,
cast(sum(Historical_Cost_of_Ride) as decimal) / sum(sum(Historical_Cost_of_Ride)) over() * 100 as [contribution %]
from dynamic_pricing
group by location_Category
order by cast(sum(Historical_Cost_of_Ride) as decimal) / sum(sum(Historical_Cost_of_Ride)) over() * 100 desc

