import os

os.environ["ABSHYD_HOME"] = "/codemill/guptshah/abshyd"
os.environ["DIRabshyd_lib_python"] = "/codemill/guptshah/abshyd/src/python"
os.environ["INFO_dataserver_override"] = "DBABS=DBABSQA,DBABSRO=DBABSQA"
os.environ["CRM_APP_INSTANCE"] = "ABS"
os.environ["abs_context"] = "dev"

import streamlit as st
from deshaw.abs.hyd.crm.service import get_events_v2
from deshaw.abs.hyd.crm.service import get_event
from deshaw.abs.hyd.crm.service import get_custom_records_for_crm_group
from deshaw.abs.hyd.crm.service import get_entities_for_user
from deshaw.abs.hyd.crm.service import get_contact_consolidated_info
from deshaw.abs.hyd.crm.service import get_organization_consolidated_info
from deshaw.abs.hyd.crm.service import get_event_comments_for_entity
from desco_llm.langchain.gateway_chat import GatewayChat
from desco_llm.commons.enums import LLMsEnum
from desco_llm.langchain.utils.utils import get_token_count_by_model
from IPython.display import HTML, Markdown, display
from langchain.schema import HumanMessage, SystemMessage
import base64
from langgraph.graph import StateGraph, START, END
from IPython.display import Image
from typing import List, Dict, Any
from langchain.tools import tool
from langchain_core.messages import AnyMessage, AIMessage, AIMessageChunk, HumanMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import StreamWriter
from langgraph.prebuilt import ToolNode
import time
from IPython.display import Image, display
from enum import StrEnum
from typing import TypedDict, List, Dict, Any
import pandas as pd
import builtins
from datetime import date

llm = GatewayChat(model="GPT-4.1-1M")
from typing import TypedDict, List, Dict, Any

class QueryState(TypedDict, total=False):
    prompt: str
    answer_log: List[str]
    tools_name: List[str]

contacts_data = get_entities_for_user(4, 1, 'guptshah')

merged_contacts_data = []
length = builtins.len(contacts_data)
print("Total Contacts: ", length)

for i in range(length):
    contact = contacts_data[i]
    contact_id = contact['entity_id']
    contact_info = get_contact_consolidated_info(4, contact_id, 'guptshah')[0]
    merged_contact_info = {**contact, **contact_info}
    merged_contacts_data.append(merged_contact_info)

df_contacts = pd.json_normalize(merged_contacts_data)
df_contacts.drop('full_names',axis =1,inplace=True)
#print(df_contacts)

columns_contacts = df_contacts.columns.tolist()

external_contact_column_descriptions = {
    "entity_id": "Unique identifier for the external contact entry.",
    "org_id": "Identifier of the organization associated with the contact.",
    "full_name": "Full name of the contact.",
    "city": "City where the contact is located.",
    "primary_email": "Primary email address of the contact.",
    "secondary_email": "Secondary or backup email address of the contact.",
    "primary_phone_number": "Main phone number used to contact the person.",
    "secondary_phone_number": "Secondary phone number for the contact.",
    "address": "Mailing or physical address of the contact.",
    "is_active": "Indicates whether the contact is currently active.",
    "description": "Description or notes about the contact.",
    "title": "Job title or designation of the contact",
    "crm_group_id": "ID linking the contact to a specific CRM group or segmentation.",
    "external_link": "URL to an external profile or resource (e.g., LinkedIn, company page).",
    "state": "State or province where the contact is located.",
    "country": "Country of the contact's location.",
    "department": "Department that the contact belongs to.",
    "first_name": "First name of the contact.",
    "middle_name": "Middle name of the contact (if applicable).",
    "last_name": "Last name or family name of the contact.",
    "nickname": "Preferred or informal name used for the contact.",
    "suffix": "Name suffix (e.g., Jr., Sr., III) of the contact.",
    "mobile_phone": "Mobile or cell phone number of the contact.",
    "other_phone": "Additional or alternative phone number for the contact.",
    "business_fax": "Fax number associated with the contact's business.",
    "notes": "Additional internal notes about the contact.",
    "child_org_id": "ID of a subsidiary or affiliated organization connected to the contact.",
    "biopic_location": "Path or URL to the contact's profile picture or bio image.",
    "org_name": "Name of the organization the contact is associated with.",
    "org_short_name": "Abbreviated version of the organization name.",
    "org_type": "General classification of the organization (e.g., client, partner).",
    "org_type_id": "Identifier for the type of the organization.",
    "event_id": "ID of a relevant event linked to the contact.",
    "last_contacted_on": "Date when the contact was last contacted.",
    "updated_on": "Date the contact was added or updated to the system.",
    "event_id_dupl": "Possible duplicate or secondary event ID for tracking.",
    "nearest_scheduled_on": "Date of the next scheduled outreach or meeting with the contact.",
    "updated_on_dupl": "Duplicate of the update timestamp, used for versioning or sync.",
    "new_last_contacted_on": "Most recent update to the 'last contacted' timestamp.",
    "priority_name": "Name or label of the priority level assigned to the contact.",
    "priority_id": "ID representing the priority level of the contact.",
    "frequency": "Expected contact frequency (e.g., monthly, quarterly).",
    "period": "Time period associated with contact frequency.",
    "org_type_name": "Descriptive name of the organization type.",
    "crm_group_id_dupl": "Duplicate of 'crm_group_id', possibly used for compatibility.",
    "using_default_priority": "Indicates whether the default priority is being used.",
    "default_user_priority": "Default priority value configured by the user.",
    "elapsed_time": "Time elapsed since the last outreach or update.",
    "is_past_frequency_configured": "Whether past outreach frequency has been set up for the contact.",
    "organization_name": "Full name of the organization associated with the contact.",
    "child_org_name": "Name of the affiliated or child organization linked to the contact.",
    "full_addresses": "All associated addresses (e.g., HQ, branches) for the contact or their org.",
    "departments": "List of departments linked to the contact or organization."
}
desco_contact_column_descriptions = {
    "desco_contact": "Primary internal (DESCO) point of contact for the external person.",
    "connected_desco_contacts": "List of internal (DESCO) users associated with the external contact.",
    "connected_desco_departments": "Departments within DESCO connected to the contact.",
    "username": "Username of the system user managing this contact.",
    "next_outreach_reminder": "Scheduled date for the next outreach reminder.",
    "is_overdue_outreach": "Indicates if the outreach to this contact is overdue.",
    "all_desco_contact": "All internal DESCO users involved with this contact.",
    "all_desco_contact_usernames": "Usernames of all DESCO contacts involved with this contact."
}

CONTACT_QUERY_PROMPT = """
    You are a contact search engine. You are given a list of contacts and their details.
    You are to filter the contacts based on the query provided in the prompt.
    The prompt is: {prompt}
    Today is {today}

    Every row in {df_contacts} contains data about two distinct groups:
    The columns in the dataframe are:
    {columns}

    External contacts: people who work outside D. E. Shaw.
    DESCO participants: D. E. Shaw employees who interacted with that external contact.

    The data for those groups live in different columns:
    External-contact columns: {external_contact_columns}
    DESCO-contact column: {desco_contacts_into}

    Never treat values from one group as belonging to the other.
    If there are multiple matches, return all of them. Same name in different organisations is not equal to same person.
    Return all contacts that satisfies the filter(s) provided in the prompt (there may be several).
    Apply user filters case-insensitively (substring OK unless exact requested).

    {{
      "contacts": [
        {{
          "full_name": "<full_name>",
          "organisation": "<organisation_name>",
          "organisation type": "<org_type or N/A>",
          "department": "department or N/A",
          "email": "<email or 'N/A'>",
          "phone": "<phone or 'N/A'>",
          "address": "<address or 'N/A'>",
          "Country": "country or N/A>",
          "date added": "<date or N/A>",
          "My last Meeting": "<last_contacted_on or N/A>",
          "Next outreach remainder": "<next outreach date or N/A>",
          "Connnected DESCO Contact(s)": "all_desco_contact",
          "Connected DESCO department(s)": "connected_desco_departments"
        }}
      ],
      "no_match": false
    }}

    If no contacts are found, return:
    {{
      "contacts": [],
      "no_match": true,
      "message": "No contacts found."
    }}
    """

from datetime import date
def get_contact_info(prompt: str):
    """
    Get contact information based on the provided prompt.

    This tool queries the CRM system's contact database and returns information about contacts
    matching the query criteria in the prompt.

    Args:
        prompt (str): The user's query about contact information.

    Returns:
        dict: Contact information matching the query in JSON format.
    """
    # Create the prompt
    contact_query_prompt = CONTACT_QUERY_PROMPT.format(
        prompt=prompt,
        today = date.today(),
        columns=columns_contacts,
        df_contacts=df_contacts.to_dict(orient='records'),
        desco_contacts_into = external_contact_column_descriptions,
        external_contact_columns = desco_contact_column_descriptions

    )
    #print(contact_query_prompt)

    # Call the LLM with the prompt
    response = llm([SystemMessage(content=contact_query_prompt), HumanMessage(content=prompt + "Note there can be multiple matches")])
    return response.content

def contact_info_tool(state: QueryState) -> QueryState:
    """
    Tool to get contact information based on the provided prompt.

    Args:
        state (QueryState)
        here, state['prompt'] is the user's query about contact information.
              state['answer_log'] is a list that stores all the previous responses.

    Returns:
        state (QueryState)
        here, state['prompt'] is the user's query about contact information.
              state['answer_log'] is a list that stores all the responses
    """
    # Get the contact information
    prompt = state['prompt']
    response = get_contact_info(prompt)

    # Update the state with the response
    state['answer_log'].append(response)
    return state

org_data = get_entities_for_user(4, 2, 'guptshah')
length = builtins.len(org_data)
#print("Total Organizations: ", length)
merged_org_data = []

for i in range(length):
    org = org_data[i]
    org_id = org['entity_id']
    org_info = get_organization_consolidated_info(4, org_id)[0]
    merged_org_info = {**org, **org_info}
    merged_org_data.append(merged_org_info)
#merged_org_data
df_org = pd.json_normalize(merged_org_data)

#df_org
columns_org = df_org.columns.tolist()
column_descriptions_org = {
    "entity_id": "Unique identifier for the organization.",
    "name": "Full legal name of the organization.",
    "address": "Primary address associated with the organization.",
    "city": "City where the organization is located.",
    "primary_email": "Main email address used to contact the organization.",
    "website": "Official website URL of the organization.",
    "fax_number": "Fax number for the organization, if available.",
    "phone_number": "Main contact phone number for the organization.",
    "crm_group_id": "Identifier linking the organization to a CRM group",
    "org_short_name": "Shortened or abbreviated version of the organization's name.",
    "state": "State or province in which the organization is located.",
    "country": "Country where the organization is based.",
    "org_type_id": "Identifier representing the type/category of the organization (e.g., client, vendor, partner).",
    "counterparty_id": "Reference ID for a related or counterparty organization, often used in financial or contractual relationships.",
    "type": "General classification of the organization (e.g., individual, company, non-profit).",
    "event_id": "ID of the most recent or relevant event associated with this organization.",
    "last_contacted_on": "Date when the organization was last contacted by any user.",
    "updated_on": "Date when the organization's record was added or updated in the system.",
    "team_last_contacted_on_event_id": "ID of the event where a team member last contacted the organization.",
    "team_last_contacted_on": "Date when any team member last contacted the organization.",
    "updated_on_dupl": "Possibly a duplicate of 'updated_on'; used in deduplication or legacy tracking.",
    "desco_contact": "Primary DESCO (internal org or system) contact for the organization.",
    "connected_desco_contacts": "List of associated DESCO team members connected to the organization.",
    "connected_desco_departments": "Internal departments of DESCO contacts connected with the organization via DESCO.",
    "full_names": "Full names of DESCO contacts associated with the organization.",
    "departments": "Internal departments of DESCO contacts connected with the organization via DESCO.",
    "entity_info": "General information or metadata about the organization.",
    "addresses": "List of all addresses associated with the organization (including alternate locations or branches)."
}
ORGANIZATION_QUERY_PROMPT = """
You are a helpful assistant that helps the user to find the organization information in the CRM system.
You are given a list of organizations with their information.
Your task is to answer the user's query based on the organization information provided.
The prompt is: {prompt}
Today is {today}.
The date format is YYYY-MM-DD.
The available columns are: {columns_org}
The columns details are: {columns_descriptions}
The dataframe name is: {df_org}

If there are multiple matches, return all of them.
Return all organisations that satisfies the filter(s) provided in the prompt (there may be several).
Apply user filters case-insensitively (substring OK unless exact requested).
IMPORTANT: Return your answer in this exact JSON format:
{{
  "organizations": [
    {{
      "name": "<organization_name>",
      "type": "<type or N/A>"
      "organization short name": "<org_short_name or N/A>",
      "phone_number": "<phone_number or 'N/A'>",
      "fax_number": "<fax number or N/A>",
      "addresses": "<addresses or 'N/A'>",
      "city": "<city or 'N/A'>",
      "state": "<state or 'N/A'>",
      "country": "<country or 'N/A'>",
      "Connected DESCO Contacts": "<full names or N/A>",
      "Department of DESCO Contacts": "<connected_desco_departments or N/A>"
    }}
  ],
  "no_match": false
}}

If no organizations are found, return:
{{
  "organizations": [],
  "no_match": true,
  "message": "No organizations found."
}}
"""

def get_organization_info(prompt: str) -> str:
    """
    Get organization information based on the provided prompt.

    This tool queries the CRM system's organization database and returns information about organizations
    matching the query criteria in the prompt.

    Args:
        prompt (str): The user's query about organization information.

    Returns:
        str: Organization information matching the query.
    """
    # Create the prompt
    org_query_prompt = ORGANIZATION_QUERY_PROMPT.format(
        prompt=prompt,
        today = date.today(),
        columns_org=columns_org,
        columns_descriptions = column_descriptions_org,
        df_org=df_org.to_dict(orient='records')
    )
    response = llm([SystemMessage(content=org_query_prompt), HumanMessage(content=prompt)])
    return response.content

def organization_info_tool(state: QueryState) -> QueryState:
    """
    Tool to get organization information based on the provided prompt.

    Args:
        state (QueryState)
        here, state['prompt'] is the user's query about organization information.
              state['answer_log'] is a list that stores all the previous responses.

    Returns:
        state (QueryState)
        here, state['prompt'] is the user's query about organization information.
              state['answer_log'] is a list that stores all the responses.
    """
    # Get the organization information
    prompt = state['prompt']
    response = get_organization_info(prompt)

    # Update the state with the response
    state['answer_log'].append(response)

    return state

df_events_details = get_events_v2(4)
df_events_details = pd.json_normalize(df_events_details)

columns_events_details = df_events_details.columns.tolist()
column_descriptions = {
    "event_id": "Unique identifier for each specific event or meeting involving internal and external contacts",
    "event_type_id": "Identifier denoting the type or category of the event",
    "updated_by": "username of the user who last updated the event details",
    "updated_on": "Date and time when the event details were last updated",
    "crm_group_id": "Identifier linking the event to a CRM group or team. Its value is 4, indicating the specific group within the CRM system",
    "event_date": "Scheduled date and time of the event",
    "created_on": "Date and time when the event record was initially created",
    "subject": "Brief title or description summarizing the event's purpose or agenda",
    "desco_contacts": "Details of internal contacts participating in the event, including name, role, and entity ID",
    "contacts": (
        "and a unique 'entity_id' that identifies the contact uniquely across the CRM system."
        "List of external contacts who attended the event from other firms. "
        "Each contact includes detailed information such as full name, the name of their organization, "
        "and their unique 'organization_entity_id' that maps to a the organization that the external contact works in"
    ),
    "has_comment": "Indicator specifying whether additional comments or notes exist for the event"
}

EVENT_QUERY_PROMPT_FOR_EVENT_DETAILS = """
You are a helpful assistant that surfaces event / meeting information from our CRM.

The user's request is:
{prompt}

Available columns:
{columns_events_details}

Column descriptions:
{column_descriptions}

DataFrame variable:
{df_events_details}

Today is {today}.
The date format is YYYY-MM-DD.

If there are multiple matches, return all of them.
Return all events that satisfies the filter(s) provided in the prompt (there may be several).
Apply user filters case-insensitively (substring OK unless exact requested).

IMPORTANT: Return your answer in this exact JSON format
(omit any field whose value is completely unavailable):

{{
  "events": [
    {{
      "event_id": <int>,
      "subject": "<string>",
      "event_date": "<YYYY-MM-DD>",
      "desco_contacts": [
        {{
          "full_name": "<string>",
          "department": "<string|null>",
          "entity_id": <int>
        }}
        // …repeat for each internal (DESCO) attendee
      ],
      "external_contacts": [
        {{
          "full_name": "<string>",
          "organization": "<string>",
          "entity_id": <int>
        }}
        // …repeat for each external attendee
      ]
    }}
    // …repeat for every matching event
  ],
  "no_match": false
}}

If **no events** match, return exactly:
{{
  "events": [],
  "no_match": true,
  "message": "No events found."
}}

Do NOT wrap the JSON in markdown back-ticks and do NOT add any commentary.
"""

import json
from bs4 import BeautifulSoup
from datetime import date, datetime, timezone

today = date.today()

def get_meeting_notes(event_id: int) -> str:
    """Generate meeting notes based on the event ID."""
    # Get the event details
    event_details = get_event(4, event_id)['comment']['comment']
    # Assuming html_content contains your HTML document
    #print(event_details + '\n')
    soup = BeautifulSoup(event_details, 'html.parser')
    meeting_notes = soup.get_text(separator="\n", strip=True)

    return meeting_notes

def get_event_info(prompt: str) -> str:
    """
    Get event information based on the provided prompt.

    This tool queries the CRM system's events database and returns information about events or meetings
    matching the query criteria in the prompt. It also summarizes meeting notes for relevant events.

    Args:
        prompt (str): The user's query about event or meeting information.

    Returns:
        str: Event information and meeting notes.
    """
    # Create the prompt
    event_query_prompt = EVENT_QUERY_PROMPT_FOR_EVENT_DETAILS.format(
        prompt=prompt,
        today=today,
        columns_events_details=columns_events_details,
        column_descriptions=column_descriptions,
        df_events_details=df_events_details.to_dict(orient='records'),
    )
    # Call the LLM with the prompt
    response = llm([SystemMessage(content=event_query_prompt), HumanMessage(content=prompt)])

    return response.content

def event_info_tool(state: QueryState) -> QueryState:
    """
    Tool to get event information based on the provided prompt.

    Args:
        state (QueryState): The current state of the query.
        state['prompt'] is the user's query about event information.
        state['answer_log'] is a list that stores all the previous responses.

    Returns:
        QueryState: The updated state with the event information and meeting notes.
        state['prompt'] is the user's query about event information.
        state['answer_log'] is a list that stores all the responses.
    """
    # Get the event information
    prompt = state['prompt']
    response = get_event_info(prompt)

    # Extract event IDs from the response
    json_text = response
    json_object = json.loads(json_text)
    #print(json_object)
    event_ids = []
    for event in json_object['events']:
        event_ids.append(event['event_id'])

    print("Event IDs: ", event_ids)
    i = 0
    # Get meeting notes for the events
    for event_id in event_ids:
        meeting_note = get_meeting_notes(event_id)
        json_object['events'][i]['meeting_notes'] = meeting_note
        i+=1

    # Update the state with the response
    state['answer_log'].append(json.dumps(json_object, indent=2))

    return state

FINALIZE_ANSWER_QUERY_PROMPT = """
You are a highly precise assistant. You are given:
User's Question:
{user_prompt}

System's Final Answer:
{final_answer}
(The system's final answer may include extra, irrelevant, or verbose information.)

Your task is to extract and present ONLY the information that directly and fully answers the user's question, formatted as clean and readable Markdown.

Instructions:
1. Read the user's question and the system’s final answer carefully.
2. Determine the entity type (Contact, Organization, Meeting/Event) and the user’s intent (e.g., filter, summary, details).
3. Follow the exact formatting and field rules below based on the entity type.
4. Never include empty, irrelevant, or unrequested fields.
5. Do not add explanations, commentary, framing text, or any extra lines before or after the output.
6. Always return the output as clean, properly formatted **Markdown**, using bullet lists or simple tables.
7. If no matching results are found, return a friendly message like:
   - `No contacts found matching your criteria.`
   - `No relevant meetings found.`
   - `No organizations match your request.`


=======================
CONTACT INFORMATION
=======================
Markdown Format:
- Use one bullet list per contact.
- Separate multiple contacts with a blank line.

Fields to include (if present):
- **Name**
- **Organization**
- **Organization Type**
- **Title**
- **Department**
- **Email**
- **Country**
- **Region**
- **Connected DESCO Contacts** (with department)

Example:
- **Name:** Amina Yusuf
    - **Organization:** PowerLight Africa
    - **Organization Type:** Private Sector
    - **Title:** Partnerships Lead
    - **Department:** Strategy
    - **Email:** amina.yusuf@powerlight.co.ke
    - **Country:** Kenya
    - **Region:** East Africa
    - **Connected DESCO Contacts:**
    - Ravi Patel (Sales)
    - Linda Zhang (Policy)

Notes:
- Do not include fields that are missing.
- Keep consistent spacing for all contacts.


=======================
ORGANIZATION DETAILS
=======================
Markdown Format:
- Use a bullet list per organization.
- Separate each with a blank line.

Fields to include (if present):
- **Organization Name**
- **Organization Type**
- **Phone Number**
- **Address** (City, State, Country)

Example:
- **Organization Name:** EcoEnergy Global
    - **Organization Type:** Non-profit
    - **Phone Number:** +1 212 555 0192
    - **Address:** Dakar, Dakar Region, Senegal

Notes:
- Omit any fields with no value.


=======================
MEETINGS / EVENTS
=======================
Markdown Format:
- Display each meeting as a grouped bullet block.
- Separate each meeting with a blank line.

If full notes are requested:
- **Meeting Subject:** Regional Solar Deployment (2025-04-12)
    - **External Contacts:** John Okoro, Anna Mehta
    - **DESCO Contacts:** Rachel Kim, Sanjay Dube
    - **Meeting Notes:**
    The team discussed project timelines, expected risks, and local compliance needs for solar rollout in West Africa.

If summary is requested:
- **Meeting Subject:** Regional Solar Deployment (2025-04-12)
    - **External Contacts:** John Okoro, Anna Mehta
    - **DESCO Contacts:** Rachel Kim, Sanjay Dube
    - **Summary:**
    - Reviewed project milestones and updated risk mitigation plans.
    - Local regulations remain a major compliance hurdle.
    - Procurement delays affecting battery supply chain.
    - Follow-up scheduled with logistics partners next week.

Notes:
- Do not summarize unless explicitly asked.
- Always show meeting subject and contacts.
- Format the summary in exactly four bullet points if applicable.


=======================
GENERAL OUTPUT RULES
=======================
- Output must be clean, properly spaced **Markdown**.
- Do not wrap in code blocks or markdown syntax (e.g., triple backticks).
- Never add any introductory or trailing text.
- If nothing matches, output only a clear message like:
  - `No contacts found matching your criteria.`
  - `No relevant meetings found.`
  - `No organizations match your request.`

Only return the clean, well-structured answer content in Markdown as described above.
"""



def finalize_answer(state: QueryState) -> QueryState:
    """
    Refines the final answer to include only the information directly relevant to the user's prompt.
    Calls the LLM with a prompt that instructs it to filter and format the answer.
    """
    user_prompt = state["prompt"]
    # Use the last answer in the log as the system's final answer
    final_answer = state["answer_log"][-1] if state["answer_log"] else ""

    st.write("Finishing up....")

    prompt = FINALIZE_ANSWER_QUERY_PROMPT.format(
        user_prompt=user_prompt,
        final_answer=final_answer
    )
    response = llm([
        SystemMessage(content=prompt)
    ])
    # Append the refined answer to the answer log
    state["answer_log"].append(response.content)
    return state


tools_description = {
    "contact_info_tool": "Get contact information based on the provided prompt. The data used by contact_info_tool are {desco_contact_column_descriptions} and {desco_contact_column_descriptions}",
    "organization_info_tool": "Get organization information based on the provided prompt. The data used by organization_info_tool is {column_descriptions_org}",
    "event_info_tool": "Get timeline or event or meeting information based on the provided prompt. The data used by event_info_tool is {column_descriptions}"
}
tool_implementations = {
    "contact_info_tool": contact_info_tool,
    "organization_info_tool": organization_info_tool,
    "event_info_tool": event_info_tool
}

ROUTER_SYSTEM = (
    "You are a tool-routing planner. Choose which agent must run next, or choose END if the user's question is fully answered. "
    "Never call the same tool twice.\n"
    "Return ONLY valid JSON in this exact format: {\"next_tool\": \"<Tool Name|END>\"}.\n"
    "Do NOT include any explanation, markdown, or extra text. Only output the JSON object."
)
ROUTER_USER_PROMPT = """
User's original question:
-------------------------
{user_query}

Partial answer so far:
----------------------
{answer}

Available Tools:
-----------------
{tools_descriptions}

Which tool should run next?
"""

#print(ToolExecutor(RouterState(prompt="Find me all event where organization Optiver was there", answer_log=[], tools_name="get_event_info")))

#print(ToolExecutor(RouterState(prompt="Find me all event where organization Optiver was there", answer_log=[], tools_name="get_event_info")))

def router(state: QueryState) -> QueryState:

    prompt = ROUTER_USER_PROMPT.format(
        user_query       = state["prompt"],
        answer           = state["answer_log"],
        tools_descriptions = tools_description,
    )

    st.write("Waiting for the router to decide the tool")

    response = llm([SystemMessage(content=ROUTER_SYSTEM),
                    HumanMessage(content=prompt)])

    print(response.content)
    next_tool = json.loads(response.content)["next_tool"].lower()
    print("Current tool:", state["tools_name"][-1])

    if next_tool not in state["tools_name"]:
        state["tools_name"].append(next_tool.lower())
    else:
        print("Tool Updated: end")
        state["tools_name"].append("end")
        st.write("No more tool calls required")

    return state


def ToolExecutor(state:QueryState) -> QueryState:
    """
    Execute the tool and return the result.
    """
    if state["tools_name"][-1]=="end":
        return state

    st.write("Running the tool")
    tool_name = state["tools_name"][-1]
    print("Executing tool:", tool_name)

    prompt = ""
    for i in state["answer_log"]:
        prompt += i
    prompt += state["prompt"]

    state["prompt"] = prompt

    # Call the appropriate function based on the tool name
    if tool_name == "contact_info_tool":
        result = contact_info_tool(state)["answer_log"][-1]
    elif tool_name == "organization_info_tool":
        result = organization_info_tool(state)["answer_log"][-1]
    elif tool_name == "event_info_tool":
        result = event_info_tool(state)["answer_log"][-1]
    else:
        raise ValueError(f"Unknown tool: {tool_name}")

    state["answer_log"].append(result)
    st.write("Sending the response back to router to check if more tool calls are required")
    return state

g = StateGraph(QueryState)

g.add_node("router", router)
g.add_node("Tool_Executor", ToolExecutor)
g.add_node("Finalize_Answer", finalize_answer)

g.add_edge(START, "router")

g.add_conditional_edges(
    "router",
    lambda state: state["tools_name"][-1] != "end",
    {
        True: "Tool_Executor",
        False: "Finalize_Answer"
    }
)

g.add_edge("Tool_Executor", "router")
g.add_edge("Finalize_Answer", END)

multiagent_executor = g.compile()

st.title("CRM Assistant")

#User Input
user_query = st.text_area("Enter your query:")

if st.button("Submit"):
    # Prepare initial state
    init_state = {
        "prompt": user_query,
        "answer_log": [],
        "tools_name": ["***"],
    }
    # Run the multi-agent executor
    final_state = multiagent_executor.invoke(init_state)
    #st.write(final_state)
    # Display the final answer
    st.markdown("### Final Answer")
    st.write(final_state["answer_log"][-1])
