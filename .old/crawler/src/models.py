from dataclasses import dataclass
from typing import Optional


@dataclass
class Article:
    id: str
    title: str
    link: str
    published: Optional[str]
    summary: Optional[str]
    content: Optional[str] = None
