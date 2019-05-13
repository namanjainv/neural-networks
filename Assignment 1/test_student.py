import pytest
import unittest

def test_name():
    from student_info import first_name, last_name
    assert first_name != "Jason"
    assert last_name != "Jennings"

def test_student_id():
    from student_info import student_id
    assert student_id != "1000123456"

def test_honor_code():
    from student_info import i_cheated_on_this_assignment
    assert i_cheated_on_this_assignment == False