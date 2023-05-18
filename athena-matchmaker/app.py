import flask as Flask
from flask import Flask, request, jsonify
import json
from flask_cors import CORS
import cohere
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct
cohere_api_key=  'XYLU0i7qUSd8okH5BeLDY7ZkfDFTnwWHJpMcWZjK'
co = cohere.Client(cohere_api_key)
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
import MySQLdb
load_dotenv()
#Steps.
#Initiate collection
#For each Mentor/Mentee, read their string data and vectorize it. Store it in the collection



qdrant_client = QdrantClient(
    url="https://9ec1c0d2-83a2-43fd-b038-17f24e63ce61.us-east-1-0.aws.cloud.qdrant.io:6333", 
    api_key="OBapBwNj1ie3igDInMu2e8SdjGXFwA33ZjxUQu--t1Azq8DmpI_L1g",
    )
    #Create Mentor Collection
qdrant_client.recreate_collection(
    collection_name="mentorVectors",
    vectors_config={
        "careerInterestList":models.VectorParams(size=1024, distance=models.Distance.COSINE),
        "hobbiesList":models.VectorParams(size=1024, distance=models.Distance.COSINE),
        },
    )
    #Create Mentee collection
qdrant_client.recreate_collection(
    collection_name="menteeVectors",
    vectors_config={
        "careerInterestList":models.VectorParams(size=1024, distance=models.Distance.COSINE),
        "hobbiesList":models.VectorParams(size=1024, distance=models.Distance.COSINE),
        },
    )





app = Flask(__name__)
CORS(app)

@app.route('/matchmake', methods=['POST'])
async def initMatchMaking():
    mentorPoints = []
   
    
    #Connect to the DB.
    connection = MySQLdb.connect(
    host= os.getenv("HOST"),
    user=os.getenv("USERNAME"),
    passwd= os.getenv("PASSWORD"),
    db= os.getenv("DATABASE"),
    ssl_mode = "VERIFY_IDENTITY",
    ssl      = {
    "ca": "/etc/ssl/cert.pem"
        }
    )
    cursor =  connection.cursor()
    #Embed ALL mentors and mentees first.
    queryMentor = "SELECT * FROM Mentor"
    queryMentee = "SELECT * FROM Mentee"
    cursor.execute(queryMentor)
    mentorData =  cursor.fetchall()
    #cursor.execute(queryMentee)
    #menteeData =  cursor.fetchall()
    careerSampleQuery = "Product management, fullstack, backend"
    hobbiesSampleQuery = "gym, biking"
    print("Attempting to embed vectors")
    for mentor in mentorData:
        careerInterests = mentor[1]
        hobbies = mentor[2]
        careerEmbeds =  co.embed(texts=[careerInterests], model = 'embed-english-light-v2.0', truncate= 'START').embeddings
        careerVectors = [float(x) for x in careerEmbeds[0]]
        hobbiesEmbeds =  co.embed(texts=[hobbies], model = 'embed-english-light-v2.0', truncate= 'START').embeddings
        hobbiesVectors = [float(x) for x in hobbiesEmbeds[0]]
        mentorPoints.append(PointStruct(id=mentor[0], vector={
            "careerInterestList":careerVectors,
            "hobbiesList":hobbiesVectors
        }, payload={"Career Value":careerInterests, "Hobbies Value":hobbies})) 
    
    operation_info = qdrant_client.upsert(
        collection_name="mentorVectors",
        wait=True,
        points = mentorPoints
    )
    print(operation_info)

    sampleCareerResponse = co.embed(texts=[careerSampleQuery], model = 'embed-english-light-v2.0', truncate= 'START').embeddings
    query_career_vector = [float(x) for x in sampleCareerResponse[0]]
    sampleHobbiesResponse = co.embed(texts=[hobbiesSampleQuery], model = 'embed-english-light-v2.0', truncate= 'START').embeddings
    query_hobbies_vector = [float(x) for x in sampleHobbiesResponse[0]]
    career_result = qdrant_client.search(
    collection_name="mentorVectors", 
    query_vector=("careerInterestList",query_career_vector),
    limit=3,
    search_params=models.SearchParams(
        exact=False
    ),
    )
    hobbies_result = qdrant_client.search(
    collection_name="mentorVectors",
    query_vector=("hobbiesList",query_hobbies_vector),
    limit=3,
    search_params=models.SearchParams(
        exact=False
    ),
    )
    print("Results of ANN with Career Interests:")
    for c in career_result:
        print(c)
    print("\n Results of ANN with Hobbies:")
    for h in hobbies_result:
        print(h)
    return json.dumps({"success":True}), 200, {"ContentType":"application/json"}


    


