
wood_list=['wood'+str(i+1)+'.jpg' for i in range(500)]

print(wood_list)

plastic_list=['pBag'+str(i+1)+'.jpg'for i in range (500)]

print(plastic_list)



    
import csv

with open('wood&plasticNew.csv', 'w', newline='') as file:
    
      writer = csv.writer(file)
     
      writer.writerow(['Image name','Image Label'])
     
      for i in wood_list:
            
          writer.writerow([i,0])
          
      for i in plastic_list:
          
          writer.writerow([i,1])
          
          
    


    
    