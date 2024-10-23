class Student:
    def __init__(self, name, student_id):
        self.name = name
        self.student_id = student_id
        self.grades = {'c语言': 0, 'Python语言': 0, '离散数学': 0}
       
    def set_grade(self, course, grade):
        for course in self.grades:
            self.grades[course] = grade

    def print_grades(self):
        print(f'学生: {self.name}    学号: {self.student_id} 的成绩为: ')
        for course in self.grades:
            print(f'{course}: {self.grades[course]} 分')

# 学生类实例化
zhang = Student ("小张","2024000111")
zhang.set_grade ("Python语言", 90)
zhang.set_grade ("离散数学", 85)

zhang = Student ("小李","2024000112")
zhang.set_grade ("Python语言", 90)
zhang.set_grade ("离散数学", 85)
zhang.print_grades ()
