# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(692, 562)
        Form.setStyleSheet("background-color: white;")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout_2.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout_2.setSpacing(10)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.painterLayout = QtWidgets.QVBoxLayout()
        self.painterLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.painterLayout.setSpacing(0)
        self.painterLayout.setObjectName("painterLayout")
        self.verticalLayout_2.addLayout(self.painterLayout)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton = QtWidgets.QPushButton(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy)
        self.pushButton.setStyleSheet("\n"
"QPushButton {\n"
"    text-decoration:none;\n"
"    text-align:center;\n"
"    padding:11px 32px;\n"
"    border: 2px solid #00c4ff;\n"
"    color:#000000;\n"
"    background:#ffffff;\n"
"    -webkit-box-shadow:0px 0px 2px #ffffff, inset 0px 0px 1px #ffffff;\n"
"    -moz-box-shadow: 0px 0px 2px #ffffff,  inset 0px 0px 1px #ffffff; \n"
"    box-shadow:0px 0px 2px #ffffff, inset 0px 0px 1px #ffffff;\n"
"}\n"
"QPushButton:hover {\n"
"    padding:11px 32px;\n"
"    border: 2px solid #00aee3;\n"
"    color:#000000;\n"
"    background:#ffffff;\n"
"    -webkit-box-shadow:0px 0px 2px #ffffff, inset 0px 0px 1px #ffffff;\n"
"    -moz-box-shadow: 0px 0px 2px #ffffff,  inset 0px 0px 1px #ffffff;\n"
"    box-shadow:0px 0px 2px #ffffff, inset 0px 0px 1px #ffffff;\n"
"}\n"
"QPushButton:pressed {\n"
"    padding:11px 32px;\n"
"    border: 2px solid #0089b3;\n"
"    color:#000000;\n"
"    background:#ffffff;\n"
"    -webkit-box-shadow:0px 0px 2px #ffffff, inset 0px 0px 1px #ffffff;\n"
"    -moz-box-shadow: 0px 0px 2px #ffffff,  inset 0px 0px 1px #ffffff; \n"
"    box-shadow:0px 0px 2px #ffffff, inset 0px 0px 1px #ffffff;\n"
"}")
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        self.label = QtWidgets.QLabel(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = QtGui.QFont()
        font.setFamily("Samsung InterFace Black")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setText("")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.verticalLayout_2.addLayout(self.horizontalLayout)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Number classifier"))
        self.pushButton.setText(_translate("Form", "Erase"))
