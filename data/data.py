import random
import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()

# إعدادات
num_students = 2500
courses_per_student = (5, 7)
terms = ['Fall 2023', 'Spring 2024', 'Fall 2024']

compulsory_subjects = [
    'HU 111', 'HU 112', 'HU 313', 'MA 111', 'MA 112', 'MA 113', 'ST 121', 'IT 111',
    'PH 111', 'CS 111', 'CS 112', 'CS 221', 'CS 214', 'CS 316', 'IT 221', 'IS 240',
    'IS 231', 'IS 211', 'IS 351', 'IT 222', 'IT 223', 'CS 241', 'CS 251', 'CS 213',
    'IS 345', 'IS 312', 'IS 313', 'IS 352', 'IS 414', 'IS 451', 'IS 333', 'IS 350',
    'IS 365', 'IS 360', 'IS 448', 'IS 498'
]

elective_subjects = [
    'HU 121', 'HU 213', 'HU 323', 'HU 331', 'HU 332', 'HU 333', 'HU 334',
    'ST 122', 'HU 122', 'IS 321', 'IT 241', 'IS 241', 'IS 342', 'IT 211',
    'CS 313', 'CS 352', 'AI 310', 'IT 423', 'IS 421', 'IS 415', 'IS 453',
    'IS 441', 'IS 442', 'IS 435', 'IS 334', 'IS 422', 'IS 416', 'IS 443',
    'IS 444', 'IS 450', 'IS 455', 'IS 460', 'IS 496', 'IS 434'
]

all_courses = compulsory_subjects + elective_subjects

# توليد طلاب ضعفاء دائمًا (10%)
weak_ratio = 0.1
weak_student_ids = set(random.sample(range(20260000, 20260000 + num_students), int(num_students * weak_ratio)))

data = []

for student_index in range(num_students):
    student_id = 20260000 + student_index
    student_name = fake.name()
    student_type = 'weak' if student_id in weak_student_ids else random.choices(['strong', 'average'], weights=[0.6, 0.4])[0]

    num_courses = random.randint(*courses_per_student)
    student_courses = random.sample(all_courses, min(num_courses, len(all_courses)))

    for course_code in student_courses:
        term = random.choice(terms)
        credits = random.choice([2, 3, 4])
        has_prerequisites = random.choices([True, False], weights=[0.8, 0.2])[0] if course_code in compulsory_subjects else random.choices([True, False], weights=[0.2, 0.8])[0]

        # توزيع الدرجات بمتوسط وانحراف معياري لكل نوع طالب
        if student_type == 'strong':
            midterm = np.random.normal(25, 5)
            final = np.random.normal(40, 7)
            assignment = np.random.normal(17, 3)
        elif student_type == 'average':
            midterm = np.random.normal(18, 7)
            final = np.random.normal(28, 8)
            assignment = np.random.normal(10, 5)
        else:  # weak
            midterm = np.random.normal(12, 6)
            final = np.random.normal(15, 7)
            assignment = np.random.normal(6, 4)

        midterm = int(np.clip(midterm, 0, 30))
        final = int(np.clip(final, 0, 50))
        assignment = int(np.clip(assignment, 0, 20))

        total = midterm + final + assignment

        # 5% خطأ في تصنيف النجاح
        if total >= 50:
            passed = True if random.random() > 0.05 else False
        else:
            passed = True if random.random() < 0.05 else False

        attendance = round(np.clip(np.random.normal(75 if student_type != 'weak' else 60, 15), 0, 100), 2)

        data.append([
            student_id, student_name, course_code, term, credits, has_prerequisites,
            midterm, final, assignment, total, passed, attendance
        ])

# إنشاء DataFrame
columns = [
    'student_id', 'student_name', 'course_code', 'term', 'credits', 'has_prerequisites',
    'midterm_score', 'final_score', 'assignment_score', 'total_score',
    'passed', 'attendance_rate'
]

df = pd.DataFrame(data, columns=columns)

# ترتيب السجلات حسب الطالب والترم
df = df.sort_values(by=['student_id', 'term']).reset_index(drop=True)

# حفظ كملف CSV
df.to_csv("student_data_realistic.csv", index=False)

print(f"تم حفظ {len(df)} سجل في student_data_realistic.csv")