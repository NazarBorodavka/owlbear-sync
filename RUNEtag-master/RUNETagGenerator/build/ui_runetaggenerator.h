/********************************************************************************
** Form generated from reading UI file 'runetaggenerator.ui'
**
** Created by: Qt User Interface Compiler version 5.15.18
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_RUNETAGGENERATOR_H
#define UI_RUNETAGGENERATOR_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QGraphicsView>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPlainTextEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_RUNETagGeneratorClass
{
public:
    QAction *actionExit;
    QAction *actionPrint;
    QAction *actionAbout;
    QAction *actionLoad_tags;
    QWidget *centralWidget;
    QVBoxLayout *verticalLayout;
    QGroupBox *groupBox;
    QHBoxLayout *horizontalLayout;
    QLabel *label;
    QSpinBox *markerSize;
    QPushButton *generateButton;
    QPushButton *nextCodeButton;
    QHBoxLayout *horizontalLayout_2;
    QLabel *label_3;
    QLabel *tagIndex;
    QLabel *label_4;
    QLineEdit *markerName;
    QSpacerItem *horizontalSpacer;
    QCheckBox *fullMarkerCheckBox;
    QCheckBox *showSlotsCheckBox;
    QCheckBox *showAxisCheckBox;
    QGraphicsView *graphicsView;
    QPlainTextEdit *crTextEdit;
    QMenuBar *menuBar;
    QMenu *menuEllyMark_Creato;
    QMenu *menuHelp;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *RUNETagGeneratorClass)
    {
        if (RUNETagGeneratorClass->objectName().isEmpty())
            RUNETagGeneratorClass->setObjectName(QString::fromUtf8("RUNETagGeneratorClass"));
        RUNETagGeneratorClass->resize(707, 728);
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(RUNETagGeneratorClass->sizePolicy().hasHeightForWidth());
        RUNETagGeneratorClass->setSizePolicy(sizePolicy);
        RUNETagGeneratorClass->setMinimumSize(QSize(520, 580));
        RUNETagGeneratorClass->setMaximumSize(QSize(99999, 99999));
        actionExit = new QAction(RUNETagGeneratorClass);
        actionExit->setObjectName(QString::fromUtf8("actionExit"));
        actionPrint = new QAction(RUNETagGeneratorClass);
        actionPrint->setObjectName(QString::fromUtf8("actionPrint"));
        actionAbout = new QAction(RUNETagGeneratorClass);
        actionAbout->setObjectName(QString::fromUtf8("actionAbout"));
        actionLoad_tags = new QAction(RUNETagGeneratorClass);
        actionLoad_tags->setObjectName(QString::fromUtf8("actionLoad_tags"));
        centralWidget = new QWidget(RUNETagGeneratorClass);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        verticalLayout = new QVBoxLayout(centralWidget);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        groupBox = new QGroupBox(centralWidget);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        QSizePolicy sizePolicy1(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(groupBox->sizePolicy().hasHeightForWidth());
        groupBox->setSizePolicy(sizePolicy1);
        groupBox->setMaximumSize(QSize(16777215, 60));
        horizontalLayout = new QHBoxLayout(groupBox);
        horizontalLayout->setSpacing(6);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        label = new QLabel(groupBox);
        label->setObjectName(QString::fromUtf8("label"));
        label->setMaximumSize(QSize(50, 16777215));

        horizontalLayout->addWidget(label);

        markerSize = new QSpinBox(groupBox);
        markerSize->setObjectName(QString::fromUtf8("markerSize"));
        markerSize->setMinimum(50);
        markerSize->setMaximum(200);
        markerSize->setValue(100);

        horizontalLayout->addWidget(markerSize);

        generateButton = new QPushButton(groupBox);
        generateButton->setObjectName(QString::fromUtf8("generateButton"));
        generateButton->setEnabled(false);

        horizontalLayout->addWidget(generateButton);

        nextCodeButton = new QPushButton(groupBox);
        nextCodeButton->setObjectName(QString::fromUtf8("nextCodeButton"));

        horizontalLayout->addWidget(nextCodeButton);


        verticalLayout->addWidget(groupBox);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        label_3 = new QLabel(centralWidget);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        horizontalLayout_2->addWidget(label_3);

        tagIndex = new QLabel(centralWidget);
        tagIndex->setObjectName(QString::fromUtf8("tagIndex"));
        tagIndex->setMinimumSize(QSize(70, 0));
        QFont font;
        font.setPointSize(11);
        font.setBold(true);
        font.setWeight(75);
        tagIndex->setFont(font);

        horizontalLayout_2->addWidget(tagIndex);

        label_4 = new QLabel(centralWidget);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        horizontalLayout_2->addWidget(label_4);

        markerName = new QLineEdit(centralWidget);
        markerName->setObjectName(QString::fromUtf8("markerName"));
        markerName->setMaximumSize(QSize(200, 16777215));
        markerName->setMaxLength(63);

        horizontalLayout_2->addWidget(markerName);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_2->addItem(horizontalSpacer);

        fullMarkerCheckBox = new QCheckBox(centralWidget);
        fullMarkerCheckBox->setObjectName(QString::fromUtf8("fullMarkerCheckBox"));
        fullMarkerCheckBox->setEnabled(true);
        fullMarkerCheckBox->setChecked(false);

        horizontalLayout_2->addWidget(fullMarkerCheckBox);

        showSlotsCheckBox = new QCheckBox(centralWidget);
        showSlotsCheckBox->setObjectName(QString::fromUtf8("showSlotsCheckBox"));

        horizontalLayout_2->addWidget(showSlotsCheckBox);

        showAxisCheckBox = new QCheckBox(centralWidget);
        showAxisCheckBox->setObjectName(QString::fromUtf8("showAxisCheckBox"));

        horizontalLayout_2->addWidget(showAxisCheckBox);


        verticalLayout->addLayout(horizontalLayout_2);

        graphicsView = new QGraphicsView(centralWidget);
        graphicsView->setObjectName(QString::fromUtf8("graphicsView"));
        sizePolicy1.setHeightForWidth(graphicsView->sizePolicy().hasHeightForWidth());
        graphicsView->setSizePolicy(sizePolicy1);
        graphicsView->setAlignment(Qt::AlignCenter);
        graphicsView->setTransformationAnchor(QGraphicsView::NoAnchor);

        verticalLayout->addWidget(graphicsView);

        crTextEdit = new QPlainTextEdit(centralWidget);
        crTextEdit->setObjectName(QString::fromUtf8("crTextEdit"));
        crTextEdit->setMaximumSize(QSize(16777215, 100));
        crTextEdit->setReadOnly(true);
        crTextEdit->setPlainText(QString::fromUtf8(""));

        verticalLayout->addWidget(crTextEdit);

        RUNETagGeneratorClass->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(RUNETagGeneratorClass);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 707, 21));
        menuEllyMark_Creato = new QMenu(menuBar);
        menuEllyMark_Creato->setObjectName(QString::fromUtf8("menuEllyMark_Creato"));
        menuHelp = new QMenu(menuBar);
        menuHelp->setObjectName(QString::fromUtf8("menuHelp"));
        RUNETagGeneratorClass->setMenuBar(menuBar);
        statusBar = new QStatusBar(RUNETagGeneratorClass);
        statusBar->setObjectName(QString::fromUtf8("statusBar"));
        RUNETagGeneratorClass->setStatusBar(statusBar);

        menuBar->addAction(menuEllyMark_Creato->menuAction());
        menuBar->addAction(menuHelp->menuAction());
        menuEllyMark_Creato->addAction(actionLoad_tags);
        menuEllyMark_Creato->addAction(actionPrint);
        menuEllyMark_Creato->addSeparator();
        menuEllyMark_Creato->addAction(actionExit);
        menuHelp->addAction(actionAbout);

        retranslateUi(RUNETagGeneratorClass);

        QMetaObject::connectSlotsByName(RUNETagGeneratorClass);
    } // setupUi

    void retranslateUi(QMainWindow *RUNETagGeneratorClass)
    {
        RUNETagGeneratorClass->setWindowTitle(QCoreApplication::translate("RUNETagGeneratorClass", "RUNETag Generator", nullptr));
        actionExit->setText(QCoreApplication::translate("RUNETagGeneratorClass", "Exit", nullptr));
        actionPrint->setText(QCoreApplication::translate("RUNETagGeneratorClass", "Export PDF", nullptr));
        actionAbout->setText(QCoreApplication::translate("RUNETagGeneratorClass", "About", nullptr));
        actionLoad_tags->setText(QCoreApplication::translate("RUNETagGeneratorClass", "Load tags", nullptr));
        groupBox->setTitle(QCoreApplication::translate("RUNETagGeneratorClass", "Marker settings", nullptr));
        label->setText(QCoreApplication::translate("RUNETagGeneratorClass", "Size (mm)", nullptr));
        generateButton->setText(QCoreApplication::translate("RUNETagGeneratorClass", "Random", nullptr));
        nextCodeButton->setText(QCoreApplication::translate("RUNETagGeneratorClass", "Next", nullptr));
        label_3->setText(QCoreApplication::translate("RUNETagGeneratorClass", "Tag IDX: ", nullptr));
        tagIndex->setText(QString());
        label_4->setText(QCoreApplication::translate("RUNETagGeneratorClass", "Name: ", nullptr));
        markerName->setText(QCoreApplication::translate("RUNETagGeneratorClass", "TestMarker", nullptr));
        fullMarkerCheckBox->setText(QCoreApplication::translate("RUNETagGeneratorClass", "Full marker", nullptr));
        showSlotsCheckBox->setText(QCoreApplication::translate("RUNETagGeneratorClass", "Show slots", nullptr));
        showAxisCheckBox->setText(QCoreApplication::translate("RUNETagGeneratorClass", "Show axis", nullptr));
        menuEllyMark_Creato->setTitle(QCoreApplication::translate("RUNETagGeneratorClass", "File", nullptr));
        menuHelp->setTitle(QCoreApplication::translate("RUNETagGeneratorClass", "Help", nullptr));
    } // retranslateUi

};

namespace Ui {
    class RUNETagGeneratorClass: public Ui_RUNETagGeneratorClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_RUNETAGGENERATOR_H
