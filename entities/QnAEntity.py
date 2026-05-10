from pydantic import BaseModel


class QnAEntity(BaseModel):
    id: int
    airline: str
    ticket_class: str
    route_type: str
    group_policy: str
    policy_type: str
    policy_desc: str
    condition_decs: str
    note: str
    applied_pax_type: str
    embedding_vector: str