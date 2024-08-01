import datetime
from pathlib import Path

from docxtpl import DocxTemplate

base_dir = Path(__file__).parent
word_template_path = base_dir / "diplomaCertificateSample.docx"
output_dir = base_dir  / "OUTPUT"
output_dir.mkdir(exist_ok=True)

today = datetime.datetime.today()
 

certification_data = [
    {"NAME": "John Doe",    "   Course": "Python Programming",                  "Date": "2024-03-01", "ID": "PY-001"},
    {"NAME": "Jane Smith",      "Course": "Data Science Fundamentals",          "Date": "2024-02-15", "ID": "DS-002"},
    {"NAME": "Michael Johnson", "Course": "Machine Learning Basics",            "Date": "2024-03-10", "ID": "ML-003"},
    {"NAME": "Emily Brown",     "Course": "Web Development with Django",        "Date": "2024-01-20", "ID": "WD-004"},
    {"NAME": "David Lee",       "Course": "Artificial Intelligence Essentials", "Date": "2024-02-05", "ID": "AI-005"},
    {"NAME": "Sarah Wilson",    "Course": "Cybersecurity Fundamentals",         "Date": "2024-04-02", "ID": "CS-006"},
    {"NAME": "Alex Clark",      "Course": "Deep Learning Fundamentals",         "Date": "2024-03-25", "ID": "DL-007"},
    {"NAME": "Emma Martinez",   "Course": "Blockchain Basics",                  "Date": "2024-01-10", "ID": "BC-008"},
    {"NAME": "William Taylor",  "Course": "Big Data Analytics",                 "Date": "2024-02-20", "ID": "BA-009"},
    {"NAME": "Olivia Rodriguez", "Course": "Cloud Computing Essentials",        "Date": "2024-03-05", "ID": "CC-010"},
    {"NAME": "Daniel Garcia",   "Course": "Internet of Things (IoT) Fundamentals", "Date": "2024-01-15", "ID": "IoT-011"},
    {"NAME": "Sophia Hernandez", "Course": "Natural Language Processing",       "Date": "2024-02-10", "ID": "NLP-012"},
    {"NAME": "Adam Thompson",   "Course": "Computer Vision Basics",             "Date": "2024-04-01", "ID": "CV-013"},
    {"NAME": "Isabella White",  "Course": "Game Development Fundamentals",      "Date": "2024-03-15", "ID": "GD-014"},
    {"NAME": "Liam Wilson",     "Course": "Robotics Essentials",                "Date": "2024-01-25", "ID": "RE-015"},
    {"NAME": "Ava Davis",       "Course": "Quantum Computing Fundamentals",     "Date": "2024-02-29", "ID": "QC-016"},
    {"NAME": "Ethan Brown",     "Course": "Augmented Reality Basics",           "Date": "2024-04-02", "ID": "AR-017"},
    {"NAME": "Mia Martinez",    "Course": "Virtual Reality Essentials",         "Date": "2024-03-20", "ID": "VR-018"},
    {"NAME": "James Johnson",   "Course": "Ethical Hacking Fundamentals",       "Date": "2024-02-05", "ID": "EH-019"},
    {"NAME": "Sophia Garcia",   "Course": "DevOps Essentials",                  "Date": "2024-01-15", "ID": "DO-020"}
]

for data in certification_data:
    # Fill the fields in the Word template with data
    doc = DocxTemplate(word_template_path) 
    doc.render(data)
    output_path = output_dir / f"{data['NAME']}.docx"  # Corrected indexing

    # Save the document with a new filename
    print("Creating ", data["NAME"]) 
    doc.save(output_path)



# # For single template
# context = {
#         "Name" : "Sagar",
#         "Course": "Python",
#         "Date": today.strftime("%Y-%m-%d"),
#         "ID":  "1234567890",
# }

# doc.render(context=context)
# doc.save(base_dir/"gen-con.docx")