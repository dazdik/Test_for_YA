import sqlite3


conn = sqlite3.connect("identifier.sqlite")
cursor = conn.cursor()

query = """
SELECT Persons.name, Positions.position
FROM Persons
INNER JOIN Positions ON Persons.pos_id = Positions.id;
"""
cursor.execute(query)

# Получение результатов
rows = cursor.fetchall()


output_file = "output.txt"


with open(output_file, "w") as file:
    file.write("name\tposition\n")
    for row in rows:
        file.write(f"{row[0]}\t{row[1]}\n")

query2 = "SELECT AVG(age) AS average_age FROM Persons WHERE age IS NOT NULL;"
cursor.execute(query2)


average_age = cursor.fetchone()[0]
print(f"The average age of the employees is: {average_age}")


query3 = "SELECT COUNT(DISTINCT name) AS unique_names_count FROM Persons WHERE name IS NOT NULL;"
cursor.execute(query3)

unique_names_count = cursor.fetchone()[0]
print(f"The amount of unique names: {unique_names_count}")


conn.close()
