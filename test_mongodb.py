from pymongo.mongo_client import MongoClient

uri = "mongodb+srv://nivakaran:QAq1jaSbJNkWLXyv@cluster0.aqkct2u.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

client = MongoClient(uri)

try:
    client.admin.command("ping")
    print("Pinged the deployment. You are successfully connected to MongoDB!")
except Exception as e:
    print(e)
