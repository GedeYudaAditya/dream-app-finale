# Database
from passlib.hash import sha256_crypt
import mysql.connector.pooling
import mysql.connector

# Configure the connection pool
dbconfig = {
    "pool_name": "mypool",
    "pool_size": 10,
    "user": "root",
    "password": "",
    "host": "localhost",
    "database": "dolphin",
}

# Create a connection pool
cnxpool = mysql.connector.pooling.MySQLConnectionPool(**dbconfig)
