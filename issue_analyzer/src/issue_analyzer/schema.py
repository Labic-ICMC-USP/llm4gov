from pydantic import BaseModel
from typing import List, Optional

class Category(BaseModel):
    label: str
    confidence: float
    keywords: List[str]

class Severity(BaseModel):
    level: str
    justification: str

class IssueType(BaseModel):
    domain: str
    subcategory: str

class IssueAnalysis(BaseModel):
    severity: Severity
    category: Category
    type: IssueType
    summary: str
    impacted_entities: List[str]
    likely_consequences: List[str]

class NamedEntities(BaseModel):
    persons: List[str]
    organizations: List[str]
    locations: List[str]
    dates: List[str]
    misc: List[str]

class Explanation(BaseModel):
    steps: List[str]
    reasoning: str

class IssueOutput(BaseModel):
    id: str
    input_text: str
    is_issue: bool
    issue_analysis: Optional[IssueAnalysis]
    named_entities: NamedEntities
    explanation: Explanation
