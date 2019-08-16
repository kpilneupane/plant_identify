indexvalue_for_database=model_out

import mysql.connector
mycursor = mydb.cursor()
CREATE DATABASE IF NOT EXISTS tempdb;
mycursor.execute("CREATE DATABASE IF NOT EXISTS Plant_identification")
mydb = mysql.connector.connect(
  host="localhost",
  user="yourusername",
  passwd="yourpassword",
  database="Plant_identification"
)
mycursor.execute("CREATE TABLE IF NOT EXISTS Plant_identification (index INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255), medicalvalue VARCHAR(255),topologicalRegion VARCHAR(255),image VARCHAR(255),availableRegion VARCHAR(255),humidity VARCHAR(255),usefulParts(255))")
name=mycursor.execute("SELECT name FROM Plant_identification where index=model_out")
s.send(name);
medicalvalue=mycursor.execute("SELECT medicalvalue FROM Plant_identification where index=model_out")
s.send(medicalvalue)
topologicalRegion=mycursor.execute("SELECT topologicalRegion FROM Plant_identification where index=model_out")
s.send(topologicalRegion)
availableRegion=mycursor.execute("SELECT availableRegion FROM Plant_identification where index=model_out")
s.send(availableRegion)
humidity=mycursor.execute("SELECT humidity FROM Plant_identification where index=model_out")
s.send(humidity)
usefulParts=mycursor.execute("SELECT usefulParts FROM Plant_identification where index=model_out")
s.send(usefulParts)
sending_image=mycursor.execute("SELECT image FROM Plant_identification where index=model_out")
s.send(sending_image)
mysql.connection.close(mydb)
