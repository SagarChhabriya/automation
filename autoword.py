import pandas as pd

# Sample certification data
certification_data = [
    {"Name": "John Doe", "Course": "Python Programming", "Date": "2024-03-01", "Certificate ID": "PY-001"},
    {"Name": "Jane Smith", "Course": "Data Science Fundamentals", "Date": "2024-02-15", "Certificate ID": "DS-002"},
    {"Name": "Michael Johnson", "Course": "Machine Learning Basics", "Date": "2024-03-10", "Certificate ID": "ML-003"},
    {"Name": "Emily Brown", "Course": "Web Development with Django", "Date": "2024-01-20", "Certificate ID": "WD-004"},
    {"Name": "David Lee", "Course": "Artificial Intelligence Essentials", "Date": "2024-02-05", "Certificate ID": "AI-005"},
    {"Name": "Sarah Wilson", "Course": "Cybersecurity Fundamentals", "Date": "2024-04-02", "Certificate ID": "CS-006"},
    {"Name": "Alex Clark", "Course": "Deep Learning Fundamentals", "Date": "2024-03-25", "Certificate ID": "DL-007"},
    {"Name": "Emma Martinez", "Course": "Blockchain Basics", "Date": "2024-01-10", "Certificate ID": "BC-008"},
    {"Name": "William Taylor", "Course": "Big Data Analytics", "Date": "2024-02-20", "Certificate ID": "BA-009"},
    {"Name": "Olivia Rodriguez", "Course": "Cloud Computing Essentials", "Date": "2024-03-05", "Certificate ID": "CC-010"},
    {"Name": "Daniel Garcia", "Course": "Internet of Things (IoT) Fundamentals", "Date": "2024-01-15", "Certificate ID": "IoT-011"},
    {"Name": "Sophia Hernandez", "Course": "Natural Language Processing", "Date": "2024-02-10", "Certificate ID": "NLP-012"},
    {"Name": "Adam Thompson", "Course": "Computer Vision Basics", "Date": "2024-04-01", "Certificate ID": "CV-013"},
    {"Name": "Isabella White", "Course": "Game Development Fundamentals", "Date": "2024-03-15", "Certificate ID": "GD-014"},
    {"Name": "Liam Wilson", "Course": "Robotics Essentials", "Date": "2024-01-25", "Certificate ID": "RE-015"},
    {"Name": "Ava Davis", "Course": "Quantum Computing Fundamentals", "Date": "2024-02-29", "Certificate ID": "QC-016"},
    {"Name": "Ethan Brown", "Course": "Augmented Reality Basics", "Date": "2024-04-02", "Certificate ID": "AR-017"},
    {"Name": "Mia Martinez", "Course": "Virtual Reality Essentials", "Date": "2024-03-20", "Certificate ID": "VR-018"},
    {"Name": "James Johnson", "Course": "Ethical Hacking Fundamentals", "Date": "2024-02-05", "Certificate ID": "EH-019"},
    {"Name": "Sophia Garcia", "Course": "DevOps Essentials", "Date": "2024-01-15", "Certificate ID": "DO-020"}
]

# Convert the data into a DataFrame
data_df = pd.DataFrame(certification_data)

# Save the DataFrame to an Excel file
data_df.to_excel("certification_data.xlsx", index=False)

print("Excel file saved successfully.")
