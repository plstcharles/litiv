#include "PlatformUtils.h"

FrameSemaphore::FrameSemaphore(int nCountMax)
	:	 m_nCount(0)
		,m_nCountMax(nCountMax) {
#if PLATFORM_USES_WIN32API
	m_oSemaphore = CreateSemaphore(NULL,0,m_nCountMax,NULL);
	if(!m_oSemaphore)
		throw std::runtime_error("Failed to create FrameSemaphore via WIN32 api");
#endif //PLATFORM_USES_WIN32API
}

FrameSemaphore::~FrameSemaphore() {
#if PLATFORM_USES_WIN32API
	CloseHandle(ghSemaphore);
#endif //PLATFORM_USES_WIN32API
}

void FrameSemaphore::notify() {
#if PLATFORM_SUPPORTS_CPP11
	std::unique_lock<std::mutex> lock(m_oMutex);
	while(m_nCount==m_nCountMax)
		m_oCondVar.wait(lock);
	++m_nCount;
	m_oCondVar.notify_one();
#elif PLATFORM_USES_WIN32API //!PLATFORM_SUPPORTS_CPP11
	m_oSemaphore
	if(!ReleaseSemaphore(m_oSemaphore,1,NULL))
		throw std::runtime_error("Failed to notify FrameSemaphore via WIN32 api");
#else //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
#error "Missing implementation for semaphores on this platform."
#endif //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
}

void FrameSemaphore::wait() {
#if PLATFORM_SUPPORTS_CPP11
	std::unique_lock<std::mutex> lock(m_oMutex);
	while(m_nCount==0)
		m_oCondVar.wait(lock);
	--m_nCount;
#elif PLATFORM_USES_WIN32API //&& !PLATFORM_SUPPORTS_CPP11
	WaitForSingleObject(m_oSemaphore,INFINITE);
#else //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
#error "Missing implementation for semaphores on this platform."
#endif //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
}

bool FrameSemaphore::try_wait() {
#if PLATFORM_SUPPORTS_CPP11
	std::lock_guard<std::mutex> lock(m_oMutex);
	if(m_nCount>0) {
		--m_nCount;
		return true;
	}
	return false;
#elif PLATFORM_USES_WIN32API //&& !PLATFORM_SUPPORTS_CPP11
	return WaitForSingleObject(m_oSemaphore,0L)==WAIT_OBJECT_0;
#else //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
#error "Missing implementation for semaphores on this platform."
#endif //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
}
