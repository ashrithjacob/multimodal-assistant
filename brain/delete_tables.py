from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import sessionmaker

engine = create_engine("postgresql+psycopg2://ashrithj:holdon@127.0.0.1:5432/cortex")

metadata = MetaData()
metadata.reflect(bind=engine)

def display_tables():
    print("Tables in the database:")
    for table in metadata.tables.values():
        print(table.name)
    print("\n")

def delete_all_tables():
    for table in metadata.tables.values():
        table.drop(engine)
    print("All tables have been dropped.")

def delete_table(table_name):
    if table_name in metadata.tables:
        metadata.tables[table_name].drop(engine)
        print(f"{table_name} has been dropped.")
    else:
        print(f"{table_name} does not exist in the database.")

if __name__ == "__main__":
    display_tables()
    table_name = input("Enter the name of the table you want to delete. Type `all` to delete all tables:")
    if table_name == "all":
        confirm = input("Are you sure you want to delete all tables? (yes/no):")
        if confirm == "yes":
            delete_all_tables()
    else:
        delete_table(table_name)
