import os
from dotenv import load_dotenv

load_dotenv()
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACKING"] = "true"

#Create a data point for evaluation
from langsmith import Client
client = Client()

# Define the dataset name and create a new dataset using the LangSmith client
dataset_name = "LangSmith Evaluation Dataset"
dataset = client.create_dataset(dataset_name)

examples=[
  {
    "input": "Who was responsible for reviewing the Jetpack Compose migration?",
    "output": "Michael Brown, the Android Tech Lead, was assigned to review the Jetpack Compose migration progress during the Android Application Performance Review meeting."
  },
  {
    "input": "Which meeting discussed PostgreSQL optimization?",
    "output": "The Database Optimization Discussion meeting on July 4, 2026 discussed PostgreSQL optimization, including indexing frequently searched columns, archiving old transactional data, and enabling query monitoring."
  },
  {
    "input": "What decisions were made to improve database performance?",
    "output": "The team decided to add indexes to frequently searched columns, archive old transactional data, and enable database query monitoring."
  },
  {
    "input": "Who was assigned to optimize SQL queries?",
    "output": "Sarah Lee was assigned to optimize the SQL queries."
  },
  {
    "input": "Which meetings focused on customer experience improvements?",
    "output": "Customer Complaints Analysis, User Experience Improvement Meeting, and Business Results Review all discussed customer experience improvements."
  },
  {
    "input": "What customer experience improvements were proposed?",
    "output": "The proposed improvements included improving search functionality, redesigning notification preferences, simplifying account management, improving onboarding, enhancing navigation, and reducing checkout steps."
  },
  {
    "input": "Which meeting discussed cloud infrastructure optimization?",
    "output": "The Cloud Infrastructure Review meeting discussed cloud infrastructure optimization."
  },
  {
    "input": "What were the cloud infrastructure decisions?",
    "output": "The team decided to remove unused environments, enable auto-scaling, and optimize cloud resource allocation."
  },
  {
    "input": "Which meeting discussed API security improvements?",
    "output": "The Security Review Meeting discussed API security improvements."
  },
  {
    "input": "What security improvements were agreed upon?",
    "output": "The team agreed to implement API rate limiting, improve authentication logging, and schedule another security audit."
  },
  {
    "input": "Which meeting discussed mobile application performance?",
    "output": "The Android Application Performance Review meeting focused on mobile application performance."
  },
  {
    "input": "What improvements were planned for the Android application?",
    "output": "The team planned to migrate remaining screens to Jetpack Compose, optimize API calls, and improve image caching."
  },
  {
    "input": "Who was responsible for configuring database monitoring alerts?",
    "output": "Mark Taylor was assigned to configure the database monitoring alerts."
  },
  {
    "input": "Which meetings mentioned analytics improvements?",
    "output": "The Product Roadmap Planning meeting and the Business Results Review discussed analytics improvements and reporting enhancements."
  },
  {
    "input": "What engineering risks were identified across the meetings?",
    "output": "Major risks included monolithic architecture scalability, slow database queries, payment API security validation, manual deployment approvals, Android performance issues, cloud infrastructure costs, and insufficient automated testing."
  },
  {
    "input": "What were the major engineering decisions across all meetings?",
    "output": "Key decisions included beginning the migration to microservices, optimizing PostgreSQL performance, completing Jetpack Compose migration, strengthening API security, automating CI/CD validation, improving automated testing, and optimizing cloud infrastructure."
  },
  {
    "input": "What action items were assigned to David Wilson?",
    "output": "No specific action items were assigned directly to David Wilson in the documented meetings."
  },
  {
    "input": "Who was responsible for reviewing campaign materials?",
    "output": "Sophia was assigned to prepare the marketing campaign materials."
  },
  {
    "input": "What was discussed during the Product Roadmap Planning meeting?",
    "output": "The meeting prioritized advanced search, notification improvements, analytics reporting, personalization features, and customer retention initiatives."
  },
  {
    "input": "Summarize the engineering priorities for the next sprint.",
    "output": "The highest priorities are database optimization, security improvements, deployment automation, payment API validation, Jetpack Compose migration, cloud cost optimization, and increased automated testing."
  }
]

client.create_examples(
    inputs=[
        {"question": ex["input"]}
        for ex in examples
    ],
    outputs=[
        {"answer": ex["output"]}
        for ex in examples
    ],
    dataset_id=dataset.id,
)

