import os
import time
import urllib.parse
from sqlalchemy import create_engine, text

def seed_database():
    """
    Creates the configured MySQL database if it doesn't exist, builds the users table,
    and seeds the default administrator account. Runs automatically on backend startup.
    """
    user = os.getenv("MYSQL_USER", "root").strip('\"\'')
    password = os.getenv("MYSQL_PASSWORD", "").strip('\"\'')
    host = os.getenv("MYSQL_HOST", "host.docker.internal").strip('\"\'')
    port = os.getenv("MYSQL_PORT", "3306").strip('\"\'')
    db_name = os.getenv("MYSQL_DATABASE", "ragbasedsql").strip('\"\'')

    admin_email = os.getenv("ADMIN_EMAIL", "admin@aeologic.com").strip('\"\'')
    admin_password = os.getenv("ADMIN_PASSWORD", "password").strip('\"\'')
    admin_name = os.getenv("ADMIN_NAME", "Aeologic User").strip('\"\'')

    print("🤖 Database Seeder: Initiating migration check...", flush=True)

    if not all([user, host, db_name]):
        print("⚠️ Database Seeder: Missing core MySQL configuration in .env. Skipping seeder.", flush=True)
        return

    # 1. Establish server-level URL (without selecting any database) to check/create target DB
    encoded_password = urllib.parse.quote_plus(password) if password else ""
    server_url = f"mysql+pymysql://{user}:{encoded_password}@{host}:{port}/"
    server_engine = create_engine(server_url)

    # 2. Connection retry loop to handle MySQL container startup latency
    connection_established = False
    for attempt in range(1, 6):
        try:
            with server_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                connection_established = True
                print(f"✅ Database Seeder: Connected to MySQL server on attempt {attempt}/5.", flush=True)
                break
        except Exception as e:
            print(f"⚠️ Database Seeder: Server connection attempt {attempt}/5 failed (MySQL might be starting up). Retrying in 3 seconds... Detail: {e}", flush=True)
            time.sleep(3)

    if not connection_established:
        print("❌ Database Seeder: Unable to connect to MySQL server after 5 attempts. Skipping seeder.", flush=True)
        return

    # 3. Create the database if it doesn't exist yet
    try:
        with server_engine.begin() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{db_name}`"))
            print(f"✅ Database Seeder: Target database '{db_name}' verified/created.", flush=True)
    except Exception as e:
        print(f"⚠️ Database Seeder: Warning during database creation: {e}", flush=True)

    # 4. Now connect directly to the target database and build the users table
    db_url = f"mysql+pymysql://{user}:{encoded_password}@{host}:{port}/{db_name}"
    db_engine = create_engine(db_url)

    create_table_query = text("""
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        email VARCHAR(255) NOT NULL UNIQUE,
        password VARCHAR(255) NOT NULL,
        name VARCHAR(255) NOT NULL,
        role VARCHAR(50) NOT NULL DEFAULT 'user'
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """)

    try:
        with db_engine.begin() as conn:
            conn.execute(create_table_query)
            print("✅ Database Seeder: 'users' table verified/created.", flush=True)

            # 5. Check if admin user already exists
            check_admin_query = text("SELECT id FROM users WHERE email = :email")
            admin_exists = conn.execute(check_admin_query, {"email": admin_email}).fetchone()

            if not admin_exists:
                insert_admin_query = text("""
                INSERT INTO users (email, password, name, role) 
                VALUES (:email, :password, :name, 'admin')
                """)
                conn.execute(insert_admin_query, {
                    "email": admin_email,
                    "password": admin_password,
                    "name": admin_name
                })
                print(f"🎉 Database Seeder: Default admin seeded successfully: {admin_email} / {admin_password}", flush=True)
            else:
                # Synchronize existing admin record's credentials with current .env configurations
                update_admin_query = text("""
                UPDATE users 
                SET password = :password, name = :name, role = 'admin' 
                WHERE email = :email
                """)
                conn.execute(update_admin_query, {
                    "email": admin_email,
                    "password": admin_password,
                    "name": admin_name
                })
                print(f"🔄 Database Seeder: Administrator credentials updated/synchronized with .env settings: {admin_email}", flush=True)

    except Exception as e:
        print(f"❌ Database Seeder: Error during table creation/seeding: {e}", flush=True)


def authenticate_user(email: str, password: str):
    """
    Authenticates a user against credentials stored in the MySQL database.
    """
    user = os.getenv("MYSQL_USER", "root").strip('\"\'')
    password_env = os.getenv("MYSQL_PASSWORD", "").strip('\"\'')
    host = os.getenv("MYSQL_HOST", "host.docker.internal").strip('\"\'')
    port = os.getenv("MYSQL_PORT", "3306").strip('\"\'')
    db_name = os.getenv("MYSQL_DATABASE", "ragbasedsql").strip('\"\'')

    encoded_password = urllib.parse.quote_plus(password_env) if password_env else ""
    db_url = f"mysql+pymysql://{user}:{encoded_password}@{host}:{port}/{db_name}"
    db_engine = create_engine(db_url)

    try:
        with db_engine.connect() as conn:
            query = text("SELECT email, password, name, role FROM users WHERE email = :email")
            res = conn.execute(query, {"email": email}).fetchone()
            if res:
                db_email, db_password, db_name_val, db_role = res
                if db_password == password:
                    return {
                        "email": db_email,
                        "name": db_name_val,
                        "role": db_role
                    }
    except Exception as e:
        print(f"❌ Auth Exception: {e}", flush=True)
    return None
