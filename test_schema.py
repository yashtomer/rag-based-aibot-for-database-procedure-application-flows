from src.database import get_engine, get_schema_summary

def test():
    print("Connecting to your database and extracting schema...")
    try:
        engine = get_engine()
        schema = get_schema_summary(engine)
        
        print("\n" + "="*50)
        print("DATABASE SCHEMA OUTPUT:")
        print("="*50 + "\n")
        
        print(schema)
        
        print("\n" + "="*50)
        print("EXTRACTION COMPLETE")
        print("="*50)
        
    except Exception as e:
        print(f"FAILED to extract schema: {e}")

if __name__ == "__main__":
    test()
