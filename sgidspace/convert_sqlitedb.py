#------------------------------------------------------------------------------
#
#  This file is part of sgidspace.
#
#  sgidspace is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  sgidspace is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with sgidspace.  If not, see <http://www.gnu.org/licenses/>.
#
#------------------------------------------------------------------------------
import sys
import random
import argparse
import zlib
import time
import random

import sqlalchemy as db
from sqlalchemy.engine import Engine
from sqlalchemy import event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import Session

@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode = MEMORY")
    cursor.execute("PRAGMA synchronous = OFF")
    cursor.close()
    return

Base = declarative_base()
class Entry(Base):
    __tablename__ = 'entries'
    i = db.Column(db.Integer, primary_key=True, autoincrement=True)
    _data = db.Column('zdata', db.Binary)

    @hybrid_property
    def data(self):
        return zlib.decompress(self._data).decode()

    @data.setter
    def data(self, x):
        self._data = zlib.compress(x.encode(), level=7)

def nice_commit(s):
    while True:
        try:
            s.commit()
            return
        except db.exc.OperationalError:
            time.sleep(random.uniform(0.1, 30))

def load_data(dbfn):
    engine = db.create_engine('sqlite:///' + dbfn)
    s = Session(bind=engine)
    Base.metadata.create_all(engine)

    for i, l in enumerate(sys.stdin.readlines()):
        l = l.strip()
        e = Entry()
        e.data = l
        s.add(e)
        
        if i % 10000 == 0:
            nice_commit(s)

    nice_commit(s)
    s.close()

    return
    
def main():
    """For converting JSONL to sqlitedb
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('sqlitedb')
    args = parser.parse_args()
    load_data(args.sqlitedb)
    return

if __name__ == '__main__':
    main()

