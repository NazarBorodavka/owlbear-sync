/****************************************************************************
** Meta object code from reading C++ file 'runetaggenerator.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.15.18)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../../../runetaggenerator.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'runetaggenerator.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.15.18. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_RUNETagGenerator_t {
    QByteArrayData data[11];
    char stringdata0[255];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_RUNETagGenerator_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_RUNETagGenerator_t qt_meta_stringdata_RUNETagGenerator = {
    {
QT_MOC_LITERAL(0, 0, 16), // "RUNETagGenerator"
QT_MOC_LITERAL(1, 17, 32), // "on_showAxisCheckBox_stateChanged"
QT_MOC_LITERAL(2, 50, 0), // ""
QT_MOC_LITERAL(3, 51, 5), // "state"
QT_MOC_LITERAL(4, 57, 33), // "on_showSlotsCheckBox_stateCha..."
QT_MOC_LITERAL(5, 91, 34), // "on_fullMarkerCheckBox_stateCh..."
QT_MOC_LITERAL(6, 126, 25), // "on_nextCodeButton_clicked"
QT_MOC_LITERAL(7, 152, 23), // "on_actionExit_triggered"
QT_MOC_LITERAL(8, 176, 24), // "on_actionAbout_triggered"
QT_MOC_LITERAL(9, 201, 28), // "on_actionLoad_tags_triggered"
QT_MOC_LITERAL(10, 230, 24) // "on_actionPrint_triggered"

    },
    "RUNETagGenerator\0on_showAxisCheckBox_stateChanged\0"
    "\0state\0on_showSlotsCheckBox_stateChanged\0"
    "on_fullMarkerCheckBox_stateChanged\0"
    "on_nextCodeButton_clicked\0"
    "on_actionExit_triggered\0"
    "on_actionAbout_triggered\0"
    "on_actionLoad_tags_triggered\0"
    "on_actionPrint_triggered"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_RUNETagGenerator[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       8,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    1,   54,    2, 0x08 /* Private */,
       4,    1,   57,    2, 0x08 /* Private */,
       5,    1,   60,    2, 0x08 /* Private */,
       6,    0,   63,    2, 0x08 /* Private */,
       7,    0,   64,    2, 0x08 /* Private */,
       8,    0,   65,    2, 0x08 /* Private */,
       9,    0,   66,    2, 0x08 /* Private */,
      10,    0,   67,    2, 0x08 /* Private */,

 // slots: parameters
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void RUNETagGenerator::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<RUNETagGenerator *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->on_showAxisCheckBox_stateChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 1: _t->on_showSlotsCheckBox_stateChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 2: _t->on_fullMarkerCheckBox_stateChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 3: _t->on_nextCodeButton_clicked(); break;
        case 4: _t->on_actionExit_triggered(); break;
        case 5: _t->on_actionAbout_triggered(); break;
        case 6: _t->on_actionLoad_tags_triggered(); break;
        case 7: _t->on_actionPrint_triggered(); break;
        default: ;
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject RUNETagGenerator::staticMetaObject = { {
    QMetaObject::SuperData::link<QMainWindow::staticMetaObject>(),
    qt_meta_stringdata_RUNETagGenerator.data,
    qt_meta_data_RUNETagGenerator,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *RUNETagGenerator::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *RUNETagGenerator::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_RUNETagGenerator.stringdata0))
        return static_cast<void*>(this);
    return QMainWindow::qt_metacast(_clname);
}

int RUNETagGenerator::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 8)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 8;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 8)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 8;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
