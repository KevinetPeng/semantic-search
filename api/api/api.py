from ninja import NinjaAPI, Schema

api = NinjaAPI()

class MatchesSchema(Schema):
    query: str
    text: str

@api.get("/matches")
def matches(request, data: MatchesSchema):
    return data