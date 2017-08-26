#include "QPBO.h"

#ifdef _MSC_VER
#pragma warning(disable: 4661)
#endif

// Instantiations

template <> 
	inline void QPBO<int>::get_type_information(char*& type_name, char*& type_format)
{
	type_name = "int";
	type_format = "d";
}

template <> 
	inline void QPBO<int64_t>::get_type_information(char*& type_name, char*& type_format)
{
	type_name = "int64_t";
	type_format = "lld";
}

template <> 
	inline void QPBO<float>::get_type_information(char*& type_name, char*& type_format)
{
	type_name = "float";
	type_format = "f";
}

template <> 
	inline void QPBO<double>::get_type_information(char*& type_name, char*& type_format)
{
	type_name = "double";
	type_format = "Lf";
}

template class QPBO<int>;
template class QPBO<int64_t>;
template class QPBO<float>;
template class QPBO<double>;
