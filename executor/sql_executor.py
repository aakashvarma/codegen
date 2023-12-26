# from sqlglot.executor import execute
#
# gpt_data_table = {
#   "table_name_64": [
#     {
#       "position": "mayor",
#       "first_election": "1988 as vice mayor 2009"
#     },
#     ...
#     {
#       "position": "mayor",
#       "first_election": "2007 as councilor 2014"
#     }
#   ]
# }
#
#  model_sql = get_llama_response(sql_prompt.format(create_table=..., query=...))
#  model_sql = model_sql[model_sql.find("<SQL>")+len("<SQL>"):model_sql.find("</SQL>")]
#  model_sql = model_sql.lower()
#
# try:
#         queryresult = execute(sql_query, tables=table)
#         modelresult = execute(model_sql, tables=table)
#         if str(queryresult) == str(modelresult):
#
#
# except Exception as e:
#         print(e)