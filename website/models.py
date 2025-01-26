from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Stock(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100))
    password = db.Column(db.String(100))
    first_name = db.Column(db.String(100))
    last_name = db.Column(db.String(100))

    stocks = db.relationship('Stock', backref='user', lazy=True)

    def add_stock(self, symbol):
        """Add a stock to user's watchlist"""
        stock = Stock(symbol=symbol, user_id=self.id)
        db.session.add(stock)
        db.session.commit()

    def remove_stock(self, symbol):
        Stock.query.filter_by(user_id=self.id, symbol=symbol).delete()
        db.session.commit()

    def get_stock_symbols(self):
        return sorted([stock.symbol for stock in self.stocks])

    def get_first_name(self):
        return self.first_name