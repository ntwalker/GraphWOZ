import numpy as np
import pandas as pd
import random
import os
import re
import sys
import base64
import time
import sqlite3
import json
import ast
import collections
from datetime import datetime, date, timedelta
from homogeneous_graph_transform import json_to_graph_indices

def triples_to_text(dialogue, triples, mappings):

    for dialogue, values in mappings.items():
        print(values.keys())

    for t in triples:
        if t[1] == "name":
            if len(t[0]) == 32: # People names
                person = t[2]
                #desc = "The name of " + t[0] + " is " + t[2]
                #group_member_relations.append(np.transpose(np.array([group, person])))
            elif len(t[0]) == 10: # Event names
                event = t[2]
                #desc = "The event " + t[0] + " is called " + t[2]
                #event_group_relations.append(np.transpose(np.array([event, group])))
        if t[1] == "email":
                desc = "The email address of " + t[0] + " is " + t[2]
        if t[1] == "phone":
                desc = "The phone number of " + t[0] + " is " + t[2]
                #print(desc)
        if t[1] == "office":
                desc = "The office of " + t[0] + " is located in room " + t[2]
                #print(desc)
        if t[1] == "attending": #Add name instead of ID
                desc =  t[0] + " is attending " + t[2]
                #print(desc)
        if t[1] == "in_group":
                desc =  t[0] + " is in the " + t[2] + " group."
                #print(desc)
        #print(t)

def relation_to_text(json_data):

    dialogue_ents, dialogue_edges, ent_name_to_id, ent_id_to_idx = json_to_graph_indices(json_data)

    docs = []
    graph_index = 1

    for dialogue in json_data:

        ent_names = []
        seen_ents = []
        prev_turn_utts = ["Now there is a converation between a visitor and a receptionist:\n"]
        graph_document = ""
        
        relevant_date = ""
        relevant_time = ""
        search_day = ""
        find_time = ""
        
        prev_agent_response = ""
        
        for idx, turn in enumerate(json_data[dialogue]["log"]):

            if not "mention" in dialogue_ents[dialogue][idx+1]["types"]:
                continue

            dt_object = datetime.fromtimestamp(turn["start_timestamp"])
            dt_tomorrow = datetime.fromtimestamp(turn["start_timestamp"]) + timedelta(days=1)
            today = str(datetime.fromtimestamp(turn["start_timestamp"]).strftime("%A, %B %d"))
            tomorrow = str(dt_tomorrow.strftime("%A, %B %d"))
            now = dt_object.strftime("%H:%M")
            soon = (dt_object + timedelta(hours=1)).strftime("%H:%M")
            
            if turn["turn"] == 1:
                prev_agent_response = "Receptionist: " + str(turn["agent_response"]) + "\n"
            else:
                prev_turn_utts.append(prev_agent_response)
                prev_agent_response = "Receptionist: " + str(turn["agent_response"]) + "\n"
                
            if type(turn["alternative"]) != float:
                prev_turn_utts.append("Visitor: " + turn["alternative"][0]["transcript"]  + "\n")
            
            #if type(turn["agent_response"]) != float and prev_agent_response != "":
                #prev_turn_utts.append(prev_agent_response)
                #prev_agent_response = turn["agent_response"]
            
            if type(turn["alternative"]) != float:
                if "request_create_event" in turn["intents"]:
                    if "@tomorrow" in turn["intents"]:
                        search_day = "tomorrow"
                        meet_time = re.findall(r"[0-9]+:[0-9]+", turn["intents"])
                        if meet_time != []:
                            hour, minute = meet_time[0].split(":")
                            find_time = dt_tomorrow.replace(hour=int(hour), minute=int(minute))
                        else:
                            find_time = dt_tomorrow
                    elif "@today" in turn["intents"]:
                        search_day = "today"
                        meet_time = re.findall(r"[0-9]+:[0-9]+", turn["intents"])
                        if meet_time != []:
                            hour, minute = meet_time[0].split(":")
                            find_time = dt_object.replace(hour=int(hour), minute=int(minute))
                        else:
                            find_time = dt_object
                    else:
                        find_time = dt_object
                        search_day = "today"
                elif "availability" in turn["intents"]:
                    search_day = "today"
                    meet_time = re.findall(r"[0-9]+:[0-9]+", turn["intents"])
                    if meet_time != []:
                        hour, minute = meet_time[0].split(":")
                        find_time = dt_object.replace(hour=int(hour), minute=int(minute))
                else:
                    find_time = "" #Reset time to search for
            
            if idx == 0:
                graph_document += "Today is " + today + ". "
                graph_document += "Tomorrow is " + tomorrow + ". "
            
            ent_mentions = turn["asr_entities"].split(", ")
            
            for e in ent_mentions:
                spl = e.split(" : ")
                if len(spl) > 1 and "@" not in spl[1]:
                    ent_names.append(spl[1])
                elif len(spl) == 1 and spl != [""]:
                    graph_document += spl[0] + " is unknown. "
                    
            for key, value in json_data[dialogue]["data"].items():
                if key in ["events"]:
                    for element in value:
                        if element["name"] in ent_names and element["name"] not in seen_ents:
                            ev_date = str(datetime.strptime(element["date"], "%Y-%m-%d").strftime("%A, %B %d"))
                            graph_document += element["name"].capitalize() + " is organized by " + element["organizer"] + " in the " + element["group"] + " group and will be held in " + element["location"] + " on " + ev_date + ". "
                            graph_document += "The meeting will start at " + str(datetime.strptime(element['start_time'], "%H:%M:%S").strftime("%I:%M")) + " and end at " + str(datetime.strptime(element['end_time'], "%H:%M:%S").strftime("%I:%M")) + ". "
                            graph_document += ", ".join(element["attendees"][:-1]) + " and " + element["attendees"][-1] + " will attend the meeting. "
                            seen_ents.append(element["name"])
                if key in ["people"]:
                    for element in value:
                        if element["name"] in ent_names and element["name"] not in seen_ents:
                            graph_document += element["name"] + " is a member of the " + element["group"] + " group. "
                            graph_document += element["name"] + "'s email is " + element["email"] + " and their phone number is " + element["phone"] + ". "
                            graph_document += element["name"] + "'s office is room " + element["office"] + ". "
                            graph_document += element["name"] + " has " + str(len(element["calendar"])) + " event(s) on his/her calendar. "
                            unavailable_now = False
                            unavailable_at_time = False
                            
                            person_today_events = []
                            person_other_events = []
                            
                            for ev in element["calendar"]:
                                event_start = ev["date"] + " " + ev["start_time"]
                                event_end = ev["date"] + " " + ev["end_time"]
                                
                                dt_event_start = datetime.strptime(event_start, "%Y-%m-%d %H:%M:%S")
                                dt_event_end = datetime.strptime(event_end, "%Y-%m-%d %H:%M:%S")
                                
                                #print(ev)
                                
                                if ev["date"] == dt_object.strftime("%Y-%m-%d"):
                                    if ev["organizer"] == element["name"]:
                                        person_today_events.append(("organizing", ev["name"], str(datetime.strptime(ev['start_time'], "%H:%M:%S").strftime("%I:%M"))))
                                        graph_document += element["name"] + " is organizing the " + ev["name"] + " today at " + ev["start_time"] + ". "
                                    else:
                                        person_today_events.append(("attending", ev["name"], str(datetime.strptime(ev['start_time'], "%H:%M:%S").strftime("%I:%M"))))
                                        graph_document += element["name"] + " will attend the " + ev["name"] + " today at " + ev["start_time"] + ". "
                                else:
                                    if ev["organizer"] == element["name"]:
                                        person_other_events.append(("organizing", ev["name"]))
                                        graph_document += element["name"] + " is organizing the " + ev["name"] + ". "
                                    else:
                                        person_other_events.append(("attending", ev["name"]))
                                        graph_document += element["name"] + " will attend the " + ev["name"] + ". "
                                
                                if len(person_today_events) > 0:
                                    c = ' '.join('is {} {} at {}'.format(*t) for t in person_today_events)
                                    #graph_document += element["name"] + " will attend the " + " ".join([mem["name"] for mem in value[k]["members"][:-1]]) + " and " + value[k]["members"][-1]["name"] + ". "
                                                            
                                if dt_event_start < dt_object < dt_event_end and unavailable_now == False:
                                    unavailable_now = True
                                    graph_document += element["name"] + " is in a meeting and not available now. "
                                
                                if type(find_time) != str:
                                    if dt_event_start < find_time < dt_event_end and unavailable_at_time == False:
                                        unavailable_at_time = True
                                        graph_document += element["name"] + " is in a meeting and not available at " + str(find_time.strftime("%I:%M")) + ". "
                                
                            if unavailable_now == False:
                                 graph_document += element["name"] + " is available and not in any meetings right now. "
                                    
                            if unavailable_at_time == False and type(find_time) != str:
                                 graph_document += element["name"] + " is available at " + str(find_time.strftime("%I:%M")) + ". "
                                
                            seen_ents.append(element["name"])
                            
                if key in ["groups"]:
                    for k in value.keys():
                        if k in ent_names and k not in seen_ents:
                            #print(k, value[k]["members"])
                            graph_document += "There are " + str(len(value[k]["members"])) + " members of the " + k + " group: " + ", ".join([mem["name"] for mem in value[k]["members"][:-1]]) + " and " + value[k]["members"][-1]["name"] + ". "
                            
                            today_events = []
                            tomorrow_events = []
                            for e in value[k]["calendar"]:
                                #print(e)
                                if e["date"] == dt_object.strftime("%Y-%m-%d"):
                                    today_events.append(e)
                                elif e["date"] == dt_tomorrow.strftime("%Y-%m-%d"):
                                    tomorrow_events.append(e)
                                    
                            if len(today_events) == 1:
                                graph_document += "The " + k + " group has " + str(len(today_events)) + " event scheduled today: " + str(today_events[0]["name"]) + ". "
                            elif len(today_events) > 1:
                                graph_document += "The " + k + " group has " + str(len(today_events)) + " events scheduled today: " + ", ".join([mem["name"] for mem in today_events[:-1]]) + " and " + today_events[-1]["name"] + ". "
                            if len(tomorrow_events) == 1:
                                graph_document += "The " + k + " group has " + str(len(tomorrow_events)) + " event scheduled tomorrow: " + str(tomorrow_events[0]["name"]) + ". "
                            elif len(tomorrow_events) > 1:
                                graph_document += "The " + k + " group has " + str(len(tomorrow_events)) + " events scheduled tomorrow: " + ", ".join([mem["name"] for mem in tomorrow_events[:-1]]) + " and " + tomorrow_events[-1]["name"] + ". "
                            #graph_document += "The " + k + " group has " + str(len(value[k]["calendar"])) + " events scheduled: " + ", ".join([mem["name"] for mem in value[k]["calendar"][:-1]]) + " and " + value[k]["calendar"][-1]["name"] + ". "
                            seen_ents.append(k)
                            
                if key in ["rooms"] and type(find_time) != str:
                    for k in value.keys():
                        if k in seen_ents:
                            continue
                        seen_ents.append(k)
                        conflict = False
                        
                        #print(find_time)
                        
                        for event in value[k]["calendar"]:
                            event_start = event["date"] + " " + event["start_time"]
                            event_end = event["date"] + " " + event["end_time"]
                            #ev_date = str(datetime.strptime(event["date"], "%Y-%m-%d").strftime("%A, %B %d"))
                            dt_event_start = datetime.strptime(event_start, "%Y-%m-%d %H:%M:%S")
                            dt_event_end = datetime.strptime(event_end, "%Y-%m-%d %H:%M:%S")
                            
                            try_event_start = dt_event_start.strftime("%H:%M")
                            
                            if dt_event_start < find_time < dt_event_end:
                                graph_document += "The " + k + " room is unavailable at " + str(find_time.strftime("%I:%M")) + " " + search_day + ". "
                                conflict = True
                                break
                        
                        if conflict == False:
                            graph_document += "The " + k + " room is available at " + str(find_time.strftime("%I:%M")) + " " + search_day + ". "
                            #print(event["date"] + " " + event["start_time"])
                            #if ev_date == today:
                            #    print(ent_mentions)
                
                        
                        #room_events = dict()
                        #for event in value[k]["calendar"]:
                        #    ev_date = str(datetime.strptime(event["date"], "%Y-%m-%d").strftime("%A, %B %d"))
                        #    if ev_date not in room_events:
                        #        room_events[ev_date] = [str(datetime.strptime(event['start_time'], "%H:%M:%S").strftime("%I:%M")) + " to " + str(datetime.strptime(event['end_time'], "%H:%M:%S").strftime("%I:%M"))]
                        #    else:
                        #        room_events[ev_date].append(str(datetime.strptime(event['start_time'], "%H:%M:%S").strftime("%I:%M")) + " to " + str(datetime.strptime(event['end_time'], "%H:%M:%S").strftime("%I:%M")))
                        #for day in room_events:
                        #    if len(room_events[day]) == 1:
                        #        graph_document += "The " + k + " room has an event from " + room_events[day][0] + " on " + day + ". "
                        #    else:
                        #        graph_document += "The " + k + " room has events from " + ", ".join([mem for mem in room_events[day][:-1]]) + " and " + room_events[day][-1] + " on " + day + ". "
                            #try:
                            #    graph_document += "The " + k + " room has " + str(len(value[k]["calendar"])) + " events scheduled: " + ", ".join([mem["name"] for mem in value[k]["calendar"][:-1]]) + " and " + value[k]["calendar"][-1]["name"] + ". "
                            #except:
                            #    print(print(k, value[k]))
                            #    break
                            #print(value[k]) 
            
            turn_document = graph_document + "\n" +  " ".join(prev_turn_utts)
            #docs.append((graph_index, idx+1, turn_document, turn["agent_response"], turn["asr_entities"]))
            docs.append((graph_index, idx+1, turn_document, turn["alternative"][0]["transcript"], turn["asr_entities"]))
            graph_index += 1
            #print(dialogue, idx+1, turn_document, turn["agent_response"])
            #print(prev_turn_utts)
            #print("\n")
        #print("\n")
    return docs

def retrieve_graph_entities(json_data):

    entities = []
    mentions = []
    mention_strings = []

    dialogue_ents, dialogue_edges, ent_name_to_id, ent_id_to_idx = json_to_graph_indices(json_data)

    for dialogue in json_data:
        for idx, turn in enumerate(json_data[dialogue]["log"]):

            if not "mention" in dialogue_ents[dialogue][idx+1]["types"]:
                continue
            else:
                ment_idx = dialogue_ents[dialogue][idx+1]["types"].index("mention")
                mns = [m for i, m in enumerate(dialogue_ents[dialogue][idx+1]["entities"]) if dialogue_ents[dialogue][idx+1]["types"][i] == "mention"]
                mention_strings.append(mns)

            turn_entities = []
            turn_mentions = []

            turn_ent_links = turn["asr_entities"].split(",")
            # noskip = 0
            # for link in turn_ent_links:
            #     if " : " in link: 
            #         mention, target_entity = link.split(" : ")
            #         if "@" in target_entity:
            #             continue
            #         else:
            #             noskip = 1
            # if noskip == 0:
            #     continue

            for entity_type in json_data[dialogue]["data"]:
                if entity_type in ["events", "people"]:
                    for ent in json_data[dialogue]["data"][entity_type]:
                        turn_entities.append(ent["name"])
                        if "office" in ent:
                            turn_entities.append(ent["office"])
                elif entity_type in ["rooms", "groups"]:
                    for ent in json_data[dialogue]["data"][entity_type]:
                        turn_entities.append(ent)
            entities.append(turn_entities)

            for link in turn_ent_links:
                if " : " in link:
                    mention, target_entity = link.split(" : ")
                    if "@" in target_entity:
                        continue
                    turn_mentions.append(turn_entities.index(target_entity.strip()))
                    #print(target_entity.strip())
            mentions.append(turn_mentions)

    return entities, mentions, mention_strings

# if __name__ == "__main__":

#     with open("graphwoz_04072022.json", "r") as file:
#         json_data = json.load(file)

#     docs = relation_to_text(json_data)
#     entities, mentions = retrieve_graph_entities(json_data)
