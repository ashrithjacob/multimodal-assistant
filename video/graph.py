import dotenv
import os
import json
from jsonschema import validate, ValidationError
from timeit import default_timer as timer
from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import AzureChatOpenAI
from openai import OpenAI
from py2neo import Graph, Node, Relationship
from neo4j import GraphDatabase
from time import sleep

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

dotenv.load_dotenv()


class TextToCypher:
    def __init__(self):
        # Instance variables to store data specific to each instance
        self.model = "gpt-3.5-turbo-1106"
        self.valid_schema = {
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "properties": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "type": {"type": "string"},
                                    "text": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "additionalProperties": True,
                            },
                        },
                        "required": ["label"],
                        "additionalProperties": False,
                    },
                },
                "relationships": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "startLabel": {"type": "string"},
                            "endLabel": {"type": "string"},
                            "properties": {
                                "type": "object",
                                "properties": {
                                    "text": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    }
                                },
                                "additionalProperties": False,
                            },
                        },
                        "required": ["type", "startLabel", "endLabel"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["nodes", "relationships"],
            "additionalProperties": False,
        }

    def description_to_first_person(self, text):
        primer = "Express the description in first person as if you were viewing it. Mention nothing else apart from what you see"
        example_1 = """
        The image appears to be taken from a first-person perspective, looking down at a smartphone being held up.
        The phone displays a picture of a woman. The background shows a room with various items, including a desk, a chair, and some clothing.
        The room seems to be a personal space, possibly a bedroom or a living room.
        """
        answer_1 = """
        I am holding up a smartphone, and the screen displays a picture of a woman.
        As I look down, I notice my surroundings—a room with items like a desk, a chair, and some clothing.
        It appears to be my personal space, maybe my bedroom or living room.
        """
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": """Express the description provided to you in first person as if you were viewing it.
                    Mention nothing else apart from what you see. assume all partially visible parts of the body to be yours""",
                },
                {"role": "user", "content": f"{primer}\n{example_1}"},
                {"role": "assistant", "content": f"{answer_1}"},
                {"role": "user", "content": f"{primer}\n{text}"},
            ],
            temperature=0.0,
        )
        first_person_response = response.choices[0].message.content
        return first_person_response

    def sentence_to_logic(self, text):
        primer = "Render the following as json, converting the text into logical units of nodes and relationships that can be difested into neo4j cypher format"
        example_1 = """In the image, I notice that it appears to be a screenshot of me during a video call or recording. 
                    I see my hands reaching upwards, possibly towards a ceiling or a wall.
                    On the wall, there is a banner or poster that reads 'LAKERS' and 'Lakers Nation'. 
                    Above my hands, I see a vent or air return on the ceiling.
                    The environment suggests that I am indoors, possibly in a room or an office
                    """
        answer_1 = """
                   {
                        "nodes": [
                            { "label": "Person", "properties": { "name": "You" } },
                            { "label": "Image", "properties": { "type": "Screenshot" } },
                            { "label": "VideoCall", "properties": { "type": "Recording" } },
                            { "label": "Hands" },
                            { "label": "CeilingOrWall" },
                            { "label": "BannerOrPoster", "properties": { "text": ["LAKERS", "Lakers Nation"] } },
                            { "label": "VentOrAirReturn" },
                            { "label": "Environment", "properties": { "type": "Indoors" } },
                            { "label": "RoomOrOffice" }
                        ],
                        "relationships": [
                            { "type": "IS_IN", "startLabel": "Person", "endLabel": "Image" },
                            { "type": "IS_A", "startLabel": "Image", "endLabel": "Screenshot" },
                            { "type": "IS_IN", "startLabel": "Person", "endLabel": "VideoCall" },
                            { "type": "REACHES_UPWARDS", "startLabel": "Person", "endLabel": "Hands" },
                            { "type": "TOWARDS", "startLabel": "Hands", "endLabel": "CeilingOrWall" },
                            { "type": "HAS", "startLabel": "CeilingOrWall", "endLabel": "BannerOrPoster" },
                            { "type": "READS", "startLabel": "BannerOrPoster", "endLabel": "Text", "properties": { "text": ["LAKERS", "Lakers Nation"] } },
                            { "type": "ABOVE", "startLabel": "VentOrAirReturn", "endLabel": "Hands" },
                            { "type": "IS_IN", "startLabel": "Person", "endLabel": "Environment" },
                            { "type": "IS_IN", "startLabel": "Environment", "endLabel": "RoomOrOffice" }
                        ]
                        }
                    """

        response = client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert in converting human text to neo4j cypher logical statements.
                            \n Ensure to capture all entities as Nodes.
                            \n Ensure to capture all relationships possible between Nodes, based on the text.
                            \n Ensure every 'person' node has the attribute name, if provided otherwise call them person_<name of person>.
                            \n In case you encounter an 'or' comsider creating a node for only one of the entitities.
                            \n Ensure all variable names are capitalised.
                            """,
                },
                {"role": "user", "content": f"{primer}:\n{example_1}"},
                {"role": "assistant", "content": f"{answer_1}"},
                {"role": "user", "content": f"{primer}:\n{text}"},
            ],
            temperature=0.0,
        )
        cypher_json = json.loads(response.choices[0].message.content)
        if self.is_valid_json(cypher_json):
            return cypher_json
        else:
            return None

    def is_valid_json(self, json_string):
        try:
            validate(json_string, self.valid_schema)
            return True
        except ValidationError:
            return False

    def cypher_json_to_query(self, cypher_json):
        cypher_query = []

        def properties_to_cypher(properties):
            return (
                "{"
                + ", ".join([f"{key}: '{value}'" for key, value in properties.items()])
                + "}"
            )

        # Process nodes
        for node in cypher_json.get("nodes", []):
            label = node.get("label")
            properties = node.get("properties", {})
            cypher_query.append(
                f"MERGE ({label.lower()}:{label} {properties_to_cypher(properties)})"
            )

        # Process relationships
        for relationship in cypher_json.get("relationships", []):
            start_label = relationship.get("startLabel")
            end_label = relationship.get("endLabel")
            rel_type = relationship.get("type")
            properties = relationship.get("properties", {})
            cypher_query.append(
                f"MERGE ({start_label.lower()})-[:{rel_type} {properties_to_cypher(properties)}]->({end_label.lower()})"
            )

        return "\n".join(cypher_query)

    def run(self, scene_description):
        first_person_description = self.description_to_first_person(scene_description)
        print(first_person_description)
        cypher_json = self.sentence_to_logic(first_person_description)
        print(cypher_json)
        cypher = self.cypher_json_to_query(cypher_json)
        return cypher


class CypherToText:
    def __init__(self):
        # OpenAI API configuration
        self.llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0,
            model_name="gpt-3.5-turbo",
        )

        # Neo4j configuration
        self.neo4j_url = os.getenv("NEO4J_URI")
        self.neo4j_user = os.getenv("NEO4J_USERNAME")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")

        self.graph = Neo4jGraph(
            url=self.neo4j_url, username=self.neo4j_user, password=self.neo4j_password
        )

    def cypher_generation__prompt(self):
        # Cypher generation prompt
        prompt = """
        You are an expert Neo4j Cypher translator who converts English to Cypher based on the Neo4j Schema provided, following the instructions below:
        1. Generate Cypher query compatible ONLY for Neo4j Version 5
        2. Do not use EXISTS, SIZE, HAVING keywords in the cypher. Use alias when using the WITH keyword
        3. Use only Nodes and relationships mentioned in the schema
        4. Always do a case-insensitive and fuzzy search for any properties related search. Eg: to search for a Client, use `toLower(client.id) contains 'neo4j'`. To search for Slack Messages, use 'toLower(SlackMessage.text) contains 'neo4j'`. To search for a project, use `toLower(project.summary) contains 'logistics platform' OR toLower(project.name) contains 'logistics platform'`.)
        5. Never use relationships that are not mentioned in the given schema
        6. When asked about projects, Match the properties using case-insensitive matching and the OR-operator, E.g, to find a logistics platform -project, use `toLower(project.summary) contains 'logistics platform' OR toLower(project.name) contains 'logistics platform'`.
        schema: {schema}
        Question: {question}
        """

        cypher_prompt = PromptTemplate(
            template=prompt, input_variables=["schema", "question"]
        )
        return cypher_prompt

    def cypher_qa_prompt(self):
        prompt = """You are an assistant that helps to form nice and human understandable answers.
        The information part contains the provided information that you must use to construct an answer.
        The provided information is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
        Make the answer sound as a response to the question. Do not mention that you based the result on the given information.
        Final answer should be easily readable and structured.
        Only respond with context provided, do not generate your own answers.
        Information:{context}
        Question: {question}
        Helpful Answer:"""

        qa_prompt = PromptTemplate(
            input_variables=["context", "question"], template=prompt
        )
        return qa_prompt

    def delete_graph(self):
        graph = Graph(password=self.neo4j_password)
        graph.delete_all()

    def display_graph_schema(self):
        self.graph.refresh_schema()
        print(self.graph.get_schema)

    def query_graph(self, user_question):
        result = None
        try:
            chain = GraphCypherQAChain.from_llm(
                llm=self.llm,
                graph=self.graph,
                verbose=True,
                return_intermediate_steps=True,
                cypher_prompt=self.cypher_generation__prompt(),
                qa_prompt=self.cypher_qa_prompt(),
            )
            result = chain.run(user_question)
            print("RESULT:\n", result)
        except Exception as e:
            print("ERROR ENCOUNTERED:\n", e)
        return result

    def inject_cypher(self, cypher):
        try:
            self.graph.query(cypher)
            print("Executed query:\n", cypher)
        except Exception as e:
            print("error:", e)


if __name__ == "__main__":
    '''
    file_path = './json/description.json'
    ttc = TextToCypher()
    # Open the file and load the JSON data
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    num = 2
    cypher = ttc.run(json_data[f"img-{num}.png"])
    print(cypher)
    '''
    ctt = CypherToText()
    # ctt.display_graph_schema()
    user_question = "provide every text that can be read in room?"
    # schema_graph()
    cypher ="""
    MERGE (person:PERSON)
    MERGE (hands:HANDS)
    MERGE (ceilingorwall:CEILINGORWALL)
    MERGE (bannerorposter:BANNERORPOSTER {text: 'LAKERS and Lakers Nation'})
    MERGE (ventorairreturn:VENTORAIRRETURN)
    MERGE (environment:ENVIRONMENT {type: 'Indoors'})
    MERGE (roomoroffice:ROOMOROFFICE)
    MERGE (person)-[:SEES]->(hands)
    MERGE (person)-[:REACHES_UPWARDS]->(hands)
    MERGE (hands)-[:TOWARDS]->(ceilingorwall)
    MERGE (ceilingorwall)-[:HAS]->(bannerorposter)
    MERGE (bannerorposter)-[:READS]->(text)
    MERGE (ventorairreturn)-[:ABOVE]->(hands)
    MERGE (person)-[:IS_IN]->(environment)
    MERGE (environment)-[:IS_IN]->(roomoroffice)
    """
    cypher2 = """
                MERGE (room:ROOM {name: "IndoorSpace"})
                MERGE (person:PERSON {name: "PERSON_1"})
                MERGE (hands:HANDS {position: "Upwards"})
                MERGE (ceiling:CEILING)
                MERGE (vent:VENT {position: "AboveHands"})
                MERGE (banner:BANNER {text: "LAKERS"})
                MERGE (poster:POSTER {text: "Lakers Nation"})
                MERGE (wall:WALL {position: "BehindPerson"})
                MERGE (room)-[:CONTAINS]->(ceiling)
                MERGE (room)-[:CONTAINS]->(wall)
                MERGE (room)-[:CONTAINS]->(banner)
                MERGE (room)-[:CONTAINS]->(poster)
                MERGE (person)-[:REACHING]->(hands)
                MERGE (person)-[:IN]->(room)
                MERGE (hands)-[:TOWARDS]->(ceiling)
                MERGE (ceiling)-[:HAS]->(vent)
                """
    #ctt.inject_cypher(cypher)
    print(ctt.query_graph(user_question))
    #ctt.delete_graph()
