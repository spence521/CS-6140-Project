#!/usr/bin/env python3
#Spencer Fronberg
#CS 6140
import math
import sys
import os
import csv

class Data(object):
    Female_Cast = 0
    Female_Crew = 0
    Trans_Cast = 0
    Trans_Crew = 0
    Ratio = 0 #total number females / total people part of movie
    Budget = 0
    Revenue = 0
    Rating = 0
    Popularity = 0
    #def __init__(self, f_cast, f_crew, trans_cast, trans_crew, ratio, budget, revenue, rating, popularity):


def Special_Split(c):
    split = True
    c_list = []
    part = ""
    quote = True
    for char in c:
        if char == "\"":
            quote = not quote
        elif char == "'" and quote:
            split = not split
        elif char == "," and split and quote:
            c_list.append(part)
            part = ""
        else:
            part = part + char
    return c_list

def Get_Counts(credit, g):
    cast = str(credit).replace("[", "")
    cast = cast.replace("]", "")
    if len(cast) > 0:
        cast = cast.replace("},", "")
        cast = cast.replace("}", "")
        cast = cast[1:]
        cast = cast.replace(": ", ":")
        cast_list = cast.split("{")
        # credit[2] #this is the ID
        gender_count = {}
        for c in cast_list:
            c_list = Special_Split(c)
            if len(c_list) > 3:
                gender = int(str(c_list[g]).split(":")[1])
                if gender in gender_count:
                    gender_count[gender] += 1
                else:
                    gender_count[gender] = 1
        return gender_count

def Get_Doc(doc):
    with open("credits.csv", "r", encoding="utf-8") as file:
        file = file.readlines()[1:]
    with open("movies_metadata.csv", "r", encoding="utf-8") as file2:
        file2 = file2.readlines()[1:]
    #with open("ratings.csv", "r", encoding="utf-8") as file3:
    #    file3 = file3.readlines()[1:]
    reader = csv.reader(file)
    reader2 = csv.reader(file2)
    #reader3 = csv.reader(file3)
    movie_meta_data = list(reader2)
    #ratings = list(reader3)
    #file.close()
    my_data = []
    my_data.append(["ID", "Female_Cast", "Female_Crew", "Transgender_Cast", "Transgender_Crew", "Male_Cast", "Male_Crew", "Total",
                    "Woman_to_Total_Ratio", "Budget", "Revenue", "Popularity", "Vote_Average", "Revenue_to_Budget_Ratio", "Budget_to_Revenue_Ratio"])
    for credit in reader:
        ID = int(credit[2])
        movie = list(i for i in movie_meta_data if str(i[5]).isdigit() and int(i[5]) == ID)[0]
        #print(ID, end="\t")
        #print(movie)
        budget = float(movie[2])
        if len(movie) > 10:
            revenue = float(movie[15])
            """try:
                runtime = float(movie[16]) / 60
            except:
                print(movie)"""
            popularity = float(movie[10])
            vote_average = float(movie[22])
            revenue_ratio = (revenue / budget if budget != 0 else 0)
            budget_ratio = (budget / revenue if revenue != 0 else 0)
        else:
            print("Continue Statement")
            continue

        cast = Get_Counts(credit[0], 3)
        crew = Get_Counts(credit[1], 2)

        f_cast = cast[1] if cast and 1 in cast else 0
        f_crew = crew[1] if crew and 1 in crew else 0
        trans_cast = cast[0] if cast and 0 in cast else 0
        trans_crew = crew[0] if crew and 0 in crew else 0
        m_cast = cast[2] if cast and 2 in cast else 0
        m_crew = crew[2] if crew and 2 in crew else 0
        total = f_cast + f_crew + trans_cast + trans_crew + m_cast + m_crew
        ratio = ((f_cast + f_crew) / (total if total != 0 else 1))
        #print("Female_Cast:\t" + str(f_cast) + "\tFemale_Crew:\t" + str(f_crew) + "\tTrans_Cast:\t" + str(trans_cast) + "\tTrans_Crew\t" +
        #      str(trans_crew) + "\tMale_Cast:\t" + str(m_cast) + "\tMale_Crew\t" + str(m_crew) + "\tTotal\t" + str(total) + "\tRatio\t" + str(ratio)
        #      + "\tBudget\t" + str(budget) + "\tRevenue\t" + str(revenue) + "\tPopularity\t" + str(popularity))
        row = [ID, f_cast, f_crew, trans_cast, trans_crew, m_cast, m_crew, total, ratio, budget, revenue, popularity, vote_average, revenue_ratio, budget_ratio]
        my_data.append(row)
    my_csv = open("New_Data.csv", "w")
    with my_csv:
        writer = csv.writer(my_csv, lineterminator='\n')
        writer.writerows(my_data)
    return

if __name__ == "__main__":
    Get_Doc("credits.csv")
    exit(0)