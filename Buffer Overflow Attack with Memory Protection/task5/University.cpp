#include "University/University.h"
#include "University/Student.h"
#include "Log/Log.h"

#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <sstream>


void University::add_student(const char *const name,
							 const char *const last_name,
							 const unsigned int id,
							 const char *const password,
							 const bool notify)
{
	// avoid double imatriculation
	std::map<const unsigned int, Student*>::const_iterator citer = student_records.find(id);
	if(citer != student_records.end())
	{
		if(notify)
		{
			std::cout << "Student with id " << id << " already imatriculated at "
					  << this->name << std::endl;

			::write_log(this, id, "Double imatriculation detected");
		}

		return;
	}

	// allocate a new Student record
	Student* record = new Student;
	record->name = new char[strlen(name) + 1];//Adding extra byte for null char
	record->last_name = new char[strlen(last_name)+ 1];//Adding extra byte for null char

	// copy students data
	record->id = id;
	strncpy(record->password,password,Student::MAX_PASSWORD_LENGTH -1);
	record->password[Student::MAX_PASSWORD_LENGTH -1] = '\0';
	//To prevent buffer overflow and ensure null termination when the destination buffer size is exactly MAX_PASSWORD_LENGTH.
	strcpy(record->name, name);
	strcpy(record->last_name, last_name);

	// append the record to the list
	student_records.insert(std::pair<const unsigned int, Student*>(id, record));

	// Notify the System about new user or just
	// loading of an existing one
	if(notify)
	{
		::write_log(this, record->id, "Student imatriculated");
		notifyStudentOnImatriculation(record->id);
	}
	else
	{
		::write_log(this, record->id, "Student loaded from Database");
	}

	return;
}

void University::initialize()
{
	// write log
	::write_log(this, 0, "Starting to load Students from Database");

	// open file stream
	std::ifstream file(db_path.c_str(), std::ifstream::in);
	if(file)
	{
		// read line by line until end of file
		while(!file.eof())
		{
			std::string line;
			std::getline(file, line);

			// extract the data from the line
			std::stringstream line_parser(line);
			std::string name;
			std::string last_name;
			std::string password;
			unsigned int id = 0;
			line_parser >> name >> last_name >> id >> password;

			if(!name.empty() && !last_name.empty() && id != 0) {
				add_student(name.c_str(), last_name.c_str(), id, password.c_str());
			}
		}

		// write log
		::write_log(this, 0, "Finished loading Students from Database");
	}
}

void University::flush()
{
	// open file stream
	std::ofstream file(db_path.c_str(), std::ifstream::out);
	if(file)
	{
		// iterate over all students
		for(std::map<const unsigned int,Student*>::const_iterator citer=student_records.begin();
			citer != student_records.end(); ++citer)
		{
			file << citer->second->name << " "
				 << citer->second->last_name << " "
				 << citer->second->id << " "
				 << citer->second->password
				 << std::endl;
		}

		// When successfully understood and solved Task 5, you
		// you will know why this call to write_log() is commented out
		// For a crazy loop, smashing the entire stack, just uncomment
		// the following lines. However, we have warned you ;-)
		///////////////////////////////////////////////////////////////
		//
		// write System Log
		//::write_log(this, 0, "Flushed Database");
	}
}

void University::close()
{
	// flush to disk
	flush();

	// free memory
	for(std::map<const unsigned int, Student*>::iterator iter = student_records.begin();
		iter != student_records.end(); ++iter)
	{
		// delete Names
		delete[] iter->second->name; iter->second->name = 0;
		delete[] iter->second->last_name; iter->second->last_name = 0;

		// delete student record
		delete iter->second;
		iter->second = 0;
	}

	// empty out map
	student_records.clear();
}

void University::print_records()
{
	// iterate over all students
	for(std::map<const unsigned int,Student*>::const_iterator citer=student_records.begin();
		citer != student_records.end(); ++citer)
	{
		// and print to console
		std::cout << citer->second->name << " "
				  << citer->second->last_name << " "
				  << citer->second->id << std::endl;
	}
}

void University::exmatriculate(const unsigned int id)
{
	std::map<const unsigned int, Student*>::iterator iter = student_records.find(id);
	if(iter != student_records.end())
	{
		std::cout << "Examtriculating Student: "
				  << iter->second->name << " "
				  << iter->second->last_name
				  << std::endl;

		// free memory
		delete[] iter->second->name; iter->second->name = 0;
		delete[] iter->second->last_name; iter->second->last_name = 0;
		delete   iter->second; iter->second = 0;
		student_records.erase(iter);

		// write out the changed data
		// to apply exmatriculations immededaly
		flush();

		// notify Student about exmatriculation
		notifyStudentOnExmatriculation(id);
	}
	else
	{
		std::cout << "No Student with id " << id
				  << " imatriculated at B-TU"
				  << std::endl;
	}
}

// Internal helper function to verify Passwords
bool check_password(const Student *const student, const char* const password)
{
	// Local copies and variables for comparison
	size_t check = 0;
	char lhs[Student::MAX_PASSWORD_LENGTH];
	char rhs[Student::MAX_PASSWORD_LENGTH];
	//------- done changes here used strncpy instead of strcpy
	strncpy ( rhs, student->password, sizeof(rhs) -1 );
	rhs[sizeof(rhs) -1] = '\0';
	strncpy ( lhs, password, sizeof(lhs) -1 );
	lhs[sizeof(lhs) -1] = '\0';
	for(size_t idx = 0; idx != Student::MAX_PASSWORD_LENGTH; ++idx)
	{
		if(lhs[idx] == '\0') // did the entered passowrd end
		{
			return rhs[idx] == '\0' ? (check == 0) : false;
		}
		else if(rhs[idx] == '\0') // did the correct password end
		{
			return lhs[idx] == '\0' ? (check == 0) : false;
		}
		else // both passwords have remaining digits to verify
		{
			check += static_cast<int>(lhs[idx] ^ rhs[idx]);
		}
	}

	return check == 0;
}

void University::request_exmatriculation(const unsigned int id, const char* const password)
{
	// Get student
	std::map<const unsigned int,Student*>::const_iterator citer = student_records.find(id);
	if(citer != student_records.end())
	{
		// check password
		if(::check_password(citer->second, password))
		{
			// remove student from db
			// Side-Effect: flush data to file
			exmatriculate(citer->first);
		}
		else
		{
			std::cout << "Invalid password!" << std::endl;
		}
	}
	else
	{
		std::cout << "No Student with id " << id
				  << " imatriculated at "  << name
				  << std::endl;
	}
}

void University::notifyStudentOnExmatriculation(const unsigned int id)
{
	// TODO: Print, sign, package, stamp and send official letter to affected students
	// NOT IMPLEMENTED YET

	std::cout << "We have sent out an exmatriculation notification to student "
			  << id
			  << "\nThank you for using B-TU Student Manager."
			  << std::endl;
}

void University::notifyStudentOnImatriculation(const unsigned int id)
{
	// TODO: Print, sign, package, stamp and send official letter to affected students
	// NOT IMPLEMENTED YET

	std::cout << "We have sent out an imatriculation letter to student "
			  << id
			  << "\nThank you for using B-TU Student Manager."
			  << std::endl;
}
