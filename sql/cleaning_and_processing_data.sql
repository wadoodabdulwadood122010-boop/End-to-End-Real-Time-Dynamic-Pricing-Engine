with encoded as (
select  
[primary_key],
      [Number_of_request],
      [Number_of_Drivers],
	  case
       when [Location_Category] = 'Urban'then 0
	   when [Location_Category] = 'Suburban'then 1
	   else 2 end as [Location_Category],

      case
       when Customer_Loyalty_Status = 'Regular'then 0
	   when Customer_Loyalty_Status = 'Silver'then 1
	   else 2 end 
	   as Customer_Loyalty_Status,

      [Number_of_Past_Rides],
      [Average_rating]
	  ,
      case
       when Time_of_boking = 'Evening'then 0
	   when Time_of_boking = 'Morning'then 1
	   when Time_of_boking = 'Aftermoon'then 2
	   else 3 end 
	   as Time_of_boking
	   ,
      case
       when Vehicle_Type = 'Premium'then 1
	   else 0
	   end as Vehicle_Type, 
      [Expected_Ride_Duration],
      [Historical_Cost_of_Ride]
from dynamic_pricing
)

select 
[primary_key]
      ,[Number_of_request]
      ,[Number_of_Drivers]
      ,[Location_Category]
      ,[Customer_Loyalty_Status]
      ,[Number_of_Past_Rides]
      ,[Average_rating]
      ,[Time_of_boking]
      ,[Vehicle_Type]
	  ,FORMAT(DATEADD(MINUTE, [Expected_Ride_Duration], 0), 'HH:mm') AS [Expected_Ride_Duration_minuts]
      ,[Historical_Cost_of_Ride]
from encoded